from typing import Any, Text, Dict, List
from numpy import disp

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from src.libs.entity_similarity import EntitySimilarity
from src.libs.helpers import load_rasa_data

LOOKUPS_PATH = 'data/lookups.yml'

# Instantiates EntitySimilarity for each entity
es = {}
es['service'] = EntitySimilarity('lev')
es['doctor'] = EntitySimilarity('jaro')

# Load lookups and rearange the in dict
lookups = load_rasa_data(LOOKUPS_PATH).lookup_tables
lookups = {lookup['name'] : lookup['elements'] for lookup in lookups}

def run_contact(
    entity_key: Text, 
    dispatcher: CollectingDispatcher, 
    tracker: Tracker) -> List[Dict[Text, Any]]:
        
        target = tracker.get_slot(entity_key)

        if target == None:        
            dispatcher.utter_message(text=f'Bonjour, j\'ai compris que vous cherchiez à contacter un {entity_key} mais je n\'ai aucune idée duquel !')
            return [SlotSet(entity_key, None)]
 
        entities = es[entity_key].get_best_entities(target, lookups[entity_key])

        if isinstance(entities, str):
            dispatcher.utter_message(text=f'Bonjour, j\'ai compris que vous cherchiez à contacter le {entity_key} {entities}!')
        else:
            entities = '\n'.join([f'\n  - {entity}' for entity in entities])
            dispatcher.utter_message(text=f'Bonjour, j\'ai compris que vous cherchiez à contacter un {entity_key} mais je ne suis pas sur duquel :\n{entities}')
    
        return [SlotSet(entity_key, None)]

class ActionContactDoctor(Action):

    def name(self) -> Text:
        return "action_contact_docor"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]):
        return run_contact('doctor', dispatcher, tracker)

class ActionContactService(Action):

    def name(self) -> Text:
        return "action_contact_service"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]):
        return run_contact('service', dispatcher, tracker)