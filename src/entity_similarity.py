from typing import Text, List, Tuple
from spacy.tokens import Doc

import spacy

MODEL_NAME = 'fr_core_news_md'

class EntitySimilarity:
    def __init__(self, spacy_model, entities : List[Text]) -> None:
        self.spacy_model = spacy_model
        # Process entities with spacy pipeline
        self.entities = [spacy_model(entity) for entity in entities]
    
    def compute_entity(self, target : Doc, entity : Doc) -> Tuple[Text, float]:
        return (str(entity), target.similarity(entity))
        
    def compute_entities(self, target : Text, sort : bool = True) -> List[Tuple[Text, float]]:
        # Process target with spacy pipeline
        target = self.spacy_model(target)
        # Compute similarity scores 
        scores = [self.compute_entity(target, entity) for entity in self.entities]
        # Sort scores if sort is true : higher score > lower score
        if sort:
            scores =  sorted(scores, key=lambda x: x[1], reverse=True)
        return scores

    def get_top_entity(self, target : Text, sort : bool = True) -> Tuple[Text, float]:
        return self.compute_entities(target, sort)[0]

    def get_top_entities(self, target : Text, n : int = 5, sort : bool = True) ->List[Tuple[Text, float]]:
        return self.compute_entities(target, sort)[:n]

if __name__ == '__main__':
    import pandas as pd
    entities = pd.read_csv('.data/entities.csv')['label'].values
    print(EntitySimilarity(spacy.load(MODEL_NAME), entities).get_top_entities('chirurgie de la main'))