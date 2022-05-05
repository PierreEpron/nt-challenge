from unittest.mock import DEFAULT
from libs.helpers import load_rasa_data
import argparse
from pathlib import Path

DEFAULT_NLU_PATH = 'data/nlu.yml'
DEFAULT_OUTPUT_PATH = 'data/nlu_converted.yml'

my_parser = argparse.ArgumentParser(
    prog='rs_to_entity',
    description='''
Convert ResponseSelector oriented nlu data file to an Entity nlu data file.

    ''',
    formatter_class=argparse.RawTextHelpFormatter
)       

my_parser.add_argument(
    '--nlu-path', '-N', action='store', default=f'{DEFAULT_NLU_PATH}', type=str,
    help=f'path to nlu data yaml file to convert. (default: {DEFAULT_NLU_PATH})')

my_parser.add_argument(
    '--output-path', '-O', action='store', default=f'{DEFAULT_OUTPUT_PATH}', type=str,
    help=f'path to save conversion result. (default: {DEFAULT_OUTPUT_PATH})')

args = my_parser.parse_args()

# nlu-path is file ?
assert Path(args.nlu_path).is_file(), f'nlu-pathis not a file : {args.nlu_path}'

data = load_rasa_data(args.nlu_path)

for example in data.training_examples:
    example.data.pop('intent_response_key', None)

data.persist_nlu(args.output_path)