from typing import Text, List, Any
import spacy
from spacy.tokens import Doc
import Levenshtein

SPACY_MODEL_NAME = 'fr_core_news_md'

def create_methods(spacy_model):
    return {
        'spacy' : lambda x, y: spacy_model(x).similarity(spacy_model(y)),
        'lev' : lambda x, y: Levenshtein.ratio(x, y),
        'jaro' : lambda x, y: Levenshtein.jaro(x, y),
        'jaro_winkler': lambda x, y: Levenshtein.jaro_winkler(x, y)
    }

class EntitySimilarity:
    def __init__(self, method_name : Text, treshold : float = .75, n : int = 5) -> None:
        # if method name is spacy load spacy model and retrieve method
        # else ensures that method name is correct and retrieve method
        if method_name == 'spacy':
            self.method = create_methods(spacy.load(SPACY_MODEL_NAME))['spacy']
        else:
            methods = create_methods(None)
            assert method_name in methods, f'You should provide a method name among those m {list(methods.keys())}.'
            self.method = methods[method_name]

        self.treshold = treshold

    def get_best_entities(self, target : Text, entities : List[Text]) -> Any:
        target = target.lower()
        scores = []
        for entity in entities:
            # Get similarity ratio
            ratio = self.method(target, entity.lower())
            
            # If ratio is 1 return directy the entity text
            # Else store entity text and ratio
            if ratio == 1:
                return entity
            else:
                scores.append((entity, ratio))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # If first entity has ratio greater than self.treshold, return first entity text
        # Else return self.n best entities text
        if scores[0][1] > self.treshold:
            return scores[0]
        return [text for text, _ in scores[:self.n]]