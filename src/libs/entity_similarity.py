from typing import Text, List, Tuple
from spacy.tokens import Doc
import Levenshtein

def create_methods(spacy_model):
    return {
        'spacy' : lambda x, y: spacy_model(x).similarity(spacy_model(y)),
        'lev' : lambda x, y: Levenshtein.ratio(x, y),
        'jaro' : lambda x, y: Levenshtein.jaro(x, y),
        'jaro_winkler': lambda x, y: Levenshtein.jaro_winkler(x, y)
    }

class EntitySimilarity:
    def __init__(self, spacy_model, entities : List[Text]) -> None:
        self.spacy_model = spacy_model
        # Process entities with spacy pipeline
        self.entities = [spacy_model(entity.lower()) for entity in entities]

    def compute_entity(self, target : Doc, entity : Doc) -> Tuple[Text, float]:
        similarity = target.similarity(entity)
        similarity = similarity if similarity != 0 else Levenshtein.jaro(str(target), str(entity))
        return (str(entity), similarity)
        
    def compute_entities(self, target : Text, sort : bool = True) -> List[Tuple[Text, float]]:
        # Process target with spacy pipeline
        target = self.spacy_model(target.lower())
        # Compute similarity scores 
        scores = [self.compute_entity(target, entity) for entity in self.entities]
        # Sort scores if sort is true : higher score > lower score
        if sort:
            scores =  sorted(scores, key=lambda x: x[1], reverse=True)
        return scores

    def get_top_entity(self, target : Text) -> Tuple[Text, float]:
        return self.compute_entities(target)[0]

    def get_top_entities(self, target : Text, n : int = 5) ->List[Tuple[Text, float]]:
        return self.compute_entities(target)[:n]

