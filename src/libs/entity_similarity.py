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
    """
    Entity Similarity object. 
    Rank value with list of others values base on similarity
    Similarity methods allowed :
        - Levenshtein (lev)
        - Jaro (jaro)
        - Jaro-winkler (jaro_winkler)
        - Word Embedding Cosinus Distance with spacy fr_core_news_md pipeline (spacy) 

    Attributes
    ----------
    method_name : str
        Name of similarity method to apply. 
        Values allowed are 'lev', 'jaro', 'jaro_winkler', 'spacy'
    treshold : str, default .75
        Treshold use for decide to return value or not for step 2 of get_main_entities
    n : int, default 5
        Number of value to return for step 3 of get_main_entities
    """

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
        self.n = n

    def get_best_entities(self, target : Text, entities : List[Text]) -> Any:
        """
            Computes similarity between target and entities.
            Return entities values following those rules :
            
            - Step 1 : return the first value from list with similarity ratio equal 1 (same value)
            - Step 2 : return all values from list with similarity ratio greater than or equal treshold
            - Step 3 : return n top values from list
        
            Parameters
            ----------
            target : str
                Text value uses for compute similarity with entities
            entities : List[str]
                Texts values uses for compute similarity with entities
            
            Return
            ------
            str or List[str]
                see rules of function

        """

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
            return scores[0][0]
        return [text for text, _ in scores[:self.n]]