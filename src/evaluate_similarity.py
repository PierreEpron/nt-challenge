from __future__ import absolute_import
from libs.helpers import load_rasa_data, get_slug
from libs.entity_similarity import create_methods
import spacy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse


DEFAULT_MODEL_NAME = 'fr_core_news_md'
DEFAULT_LOOKUPS_PATH = '.data/nlu_0.yml'
DEFAULT_EXAMPLES_PATH = 'tests/nlu.yml'
DEFAULT_ROLE = 'target'
DEFAULT_OUTPUT_DIR = 'results/'
DEFAULT_BEST_METHOD_CRITERIA = 'mean'

REPORT_FILE_NAME = 'similarity_report.json'
BEST_METHOD_CRITERIAS = ['mean', 'min', 'max', 'all']

#region argparse

my_parser = argparse.ArgumentParser(
    prog='evaluate_similarity',
    description='''
Evaluate 4 methods of similarity on nlu examples.

Similarity methods :
    - Levenshtein
    - Jaro
    - Jaro-winkler
    - Word Embedding Cosinus Distance with spacy
    
Use entities values from nlu examples as first string.
Use each value in lookup tables as second string. 

Entities should have the same name than a lookup table to be use. 
If --role is provided, only entity with this role will be use. 

Intent of example will be use for compute ranking of entity. Example :
    intent 'contact_doctor/x' will be parsed as x
    slugify(entity) == x will be use for ranking

Store report result in ouput-dir/{REPORT_FILE_NAME}.
Print best method based on best-method-criteria (less is better)
    ''',
    formatter_class=argparse.RawTextHelpFormatter
)       

my_parser.add_argument(
    '--model-name', '-M',  action='store', default=f'{DEFAULT_MODEL_NAME}', type=str,
    help=f'The spacy pipeline to load. (default: {DEFAULT_MODEL_NAME})')

my_parser.add_argument(
    '--lookups-path', '-L', action='store', default=f'{DEFAULT_LOOKUPS_PATH}', type=str,
    help=f'The yaml file use to find lookup tables. (default: {DEFAULT_LOOKUPS_PATH})')

my_parser.add_argument(
    '--examples-path', '-E', action='store', default=f'{DEFAULT_EXAMPLES_PATH}', type=str,
    help=f'The yaml file use to find nlu examples. (default: {DEFAULT_EXAMPLES_PATH})')

my_parser.add_argument(
    '--role', '-R', action='store', default=f'{DEFAULT_ROLE}', type=str,
    help=f'Role use for filter entities. (default: {DEFAULT_ROLE})')

my_parser.add_argument(
    '--output-dir', '-O', action='store', default=f'{DEFAULT_OUTPUT_DIR}', type=str,
    help=f'directory used for store result. (default: {DEFAULT_OUTPUT_DIR})')

my_parser.add_argument(
    '--best-method-criteria', '-B', action='store', default=f'{DEFAULT_BEST_METHOD_CRITERIA}', type=str,
    help=f'value used for choose best method ({BEST_METHOD_CRITERIAS}) (default: {DEFAULT_BEST_METHOD_CRITERIA})')

args = my_parser.parse_args()

#endregion

#region validate args

# is lookups-path a file ?
assert Path(args.lookups_path).is_file(), f'lookups-path is not a file : {args.lookups_path}'

# is examples-path a file ?
assert Path(args.examples_path).is_file(), f'examples-path is not a file : {args.examples_path}'

# is best-method-criteria in mean, min, max, all ?
assert args.best_method_criteria in BEST_METHOD_CRITERIAS, f'best-method-criteria should be one of this options : {BEST_METHOD_CRITERIAS}'

output_dir = Path(args.output_dir)
if not output_dir.is_dir():
    output_dir.mkdir()

#endregion

#region helpers

def create_score_dict():
    scrore_dict = {'mean': 0, 'min':0, 'max':0, 'values':[]}
    return {
        'rank': dict(scrore_dict),
        'ratio': dict(scrore_dict),
        'not_found': 0
    }

def has_role(role):
    def wrapped_has_role(entity):
        return 'role' in entity and entity['role'] == role
    return wrapped_has_role if role != '' else lambda x: True

def is_intent(intent):
    def wrapped_is_intent(entity):
        return get_slug(entity[1][0]) == intent.split('/')[1]
    return wrapped_is_intent

def compute_similarities(target, entities, method):
        target = target.lower()
        # Compute similarity scores 
        scores = []
        tot = 0
        for entity in entities:
            s = method(target, entity.lower())
            tot += s
            scores.append((entity, s))
        return [] if tot == 0 else sorted(scores, key=lambda x: x[1], reverse=True)

def set_score_stats(score, values):
    if len(values) > 0:
        score['values'] = values
        score['mean'] = float(np.mean(values))
        score['min'] = float(np.min(values))
        score['max'] = float(np.max(values))

def report_for_criteria(criteria):
    for k_lookup, lookup in scores.items():
        best_method = None
        for k_method, method in lookup.items():
            if best_method == None:
                best_method = method
            elif method['rank'][criteria] < best_method['rank'][criteria]:
                best_method = method
        print(f'''
    {k_lookup} best method : {k_method} ({criteria})
        rank : 
            mean : {best_method['rank']['mean']}
            min : {best_method['rank']['min']}
            max : {best_method['rank']['max']}
        ratio :
            mean : {best_method['ratio']['mean']}
            min : {best_method['ratio']['min']}
            max : {best_method['ratio']['max']}
        not_found: {best_method['not_found']}''')            
    return best_method

#endregion

#region init

# Load spacy pipeline
spacy_model = spacy.load(args.model_name)

# Load lookups and rearange the in dict
lookups = load_rasa_data(args.lookups_path).lookup_tables
lookups = {lookup['name'] : lookup['elements'] for lookup in lookups}

# Load examples
examples = load_rasa_data(args.examples_path).entity_examples

# Define methods
methods = create_methods(spacy_model)

#endregion

#region main loop

scores = {}

# for each lookup, method
for k_lookup, lookup in lookups.items():
    scores[k_lookup] = {}
    for k_method, method in methods.items():
        # create full score dict for lookup and method
        scores[k_lookup][k_method] = score = create_score_dict()
        # list for keep values and ratios during process
        rank_values = []
        ratio_values = []

        # for each example and filtered entities
        for example in examples:
            for entity in filter(
                lambda x: x['entity'] == k_lookup and has_role(args.role)(x), 
                example.data['entities']):

                value = entity['value']
                intent = example.data['intent_response_key']

                # Get ranking of intent in entity similarities 
                # rank = (rank, (entity_value, similarity_ratio))
                # return none if intent not found or if all similarities score equals 0
                rank = next(filter(is_intent(intent), 
                       # Enumerate similarities for get ranking
                        enumerate(compute_similarities(value, lookups[k_lookup], method))),
                        None)

                # if there rank found, keep rank and similarity_ratio
                # else add 1 to not_found count
                if rank:
                    rank_values.append(rank[0])
                    ratio_values.append(rank[1][1])
                else:
                    score['not_found'] += 1

        # compute min max mean for rank and ratio
        set_score_stats(score['rank'], rank_values)
        set_score_stats(score['ratio'], ratio_values)

#endregion

#region report

if args.best_method_criteria == 'all':
    for criteria in ['mean', 'min', 'max']:
        report_for_criteria(criteria)
else:
    best_method = report_for_criteria(args.best_method_criteria)

(output_dir / REPORT_FILE_NAME).write_text(json.dumps(scores), encoding='utf-8')

#endregion

