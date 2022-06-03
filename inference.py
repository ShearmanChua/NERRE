import time
import requests
from typing import List, Dict, Tuple

import pandas as pd
import json
import torch
import flair
import opennre
from flair.data import Sentence
from flair.models import SequenceTagger

flair.device = torch.device('cpu')

def load_tagger(name: str) -> SequenceTagger:
    return SequenceTagger.load(name)

def load_re_model(name: str, device: str = 'cuda') -> opennre.model.SoftmaxNN:
    model = opennre.get_model(name)
    if device == 'cuda':
        model = model.cuda()
    return model

def predict_el_templates(dataset: List[Dict]):
    response = requests.post('http://entitylinking:5000/entity_linker_point', json = dataset)
    #ipdb.set_trace()
    print([doc for doc in response.iter_lines()])
    response = response.json()
    return response

def generate_named_entities(tagger: SequenceTagger, text: str) -> List[Dict[str, str]]:
    # make Flair sentence
    sentence = Sentence(text)
    # predict NER tags
    tagger.predict(sentence)

    entities, el_temps = [], []
    # print predicted NER spans
    # print('The following NER tags are found:')
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        # print(entity)
        ent = {}
        start_pos = entity.tokens[0].start_pos
        end_pos = entity.tokens[-1].end_pos

        ent['start_pos'] = start_pos
        ent['end_pos'] = end_pos
        ent['text'] = text[start_pos:end_pos]
        ent['type'] = entity.tag

        entities.append(ent)
        el_temps.append({
            'left_context': text[:start_pos],
            'mention': ent['text'],
            'right_context': text[end_pos:]
        })

    return entities, el_temps

def extract_linked_entities(
    el_candidates: List[str], 
    entities: List[Dict[str, str]]) -> List[Dict[str, str]]:

    linked = []
    for cand, ent in zip(el_candidates, entities):
        if cand is not None:
            ent['node_id'] = cand
            linked.append(ent)
    
    return linked

def generate_head_tail_pairs(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:

    head_tail_pairs = []

    for head_idx in range(0,len(entities)):
        head_span = (entities[head_idx]['start_pos'],entities[head_idx]['end_pos'])
        head_text = entities[head_idx]['text']
        head_id = entities[head_idx]['node_id']
        for tail_idx in range(head_idx+1,len(entities)):
            tail_span = (entities[tail_idx]['start_pos'],entities[tail_idx]['end_pos'])
            tail_text = entities[tail_idx]['text']
            tail_id = entities[head_idx]['node_id']
            if head_text != tail_text:
                head_tail_pairs.append({'head_span':head_span,'tail_span':tail_span,'head_text':head_text,'tail_text':tail_text, 'head_id': head_id, 'tail_id': tail_id})

    return head_tail_pairs

def extract_relations(text: str, relation_json, re_model: opennre.model.SoftmaxNN, head_tail_pairs: List[Dict[str, str]], confidence_threshold: float = 0.7) -> List[Tuple[str]]:

    with open(relation_json, 'r') as f:
        relation_to_idx = json.load(f)

    relations = []

    for pair in head_tail_pairs:
        relation, confidence = re_model.infer({'text': text, 'h': {'pos': pair['head_span']}, 't': {'pos': pair['tail_span']}})
        if confidence > confidence_threshold:
            relations.append({'subject': pair['head_id'], 'relation': relation_to_idx[relation], 'object': pair['tail_id']})
    
    return relations

def generate_ner_query_output(entities: Dict[str, str]):

    output = {'entities': {}}
    for ent in entities:
        entity_span = str(ent['start_pos']) + '_' + str(ent['end_pos'])
        node_id = ent['node_id']
        node_type = ent['type']
        output['entities'][entity_span] = {'node_id': node_id, 'node_type': node_type}

    return output

def do_everything(text: Dict[str, str]):
    # load tagger
    tagger = load_tagger("flair/ner-english-large")
    re_model = load_re_model('wiki80_bert_softmax', device='cpu')
    json_path = 'wiki80_rel2id.json'

    start1 = time.time()

    entities, el_templates = generate_named_entities(tagger, text['sentence'])

    end1 = time.time()
    print('Time taken for NER:', end1-start1)

    top_candidate_list, resolved_entities = predict_el_templates(el_templates)

    end2 = time.time()
    print('Time taken for EL:', end2-end1)
    print(resolved_entities)

    linked_entities = extract_linked_entities(top_candidate_list, entities)
    head_tail_pairs = generate_head_tail_pairs(linked_entities)

    end3 = time.time()
    print('time taken for generating head-tail pairs:', end3-end2)

    relations = extract_relations(text['sentence'], json_path, re_model, head_tail_pairs)

    end4 = time.time()
    print('time taken for RE', end4-end3)

    return linked_entities, relations

def generate_neo4j_dfs(relation_triplets, relation_json, entities):

    with open(relation_json, 'r') as f:
        relation_to_idx = json.load(f)

    relation_list = []
    for key in relation_to_idx.keys():
        relation_list.append({
            'relation': key,
            'relation_id': relation_to_idx[key]}
        )
    
    relation_df = pd.DataFrame(relation_list)

    # nodes_df = pd.DataFrame(columns=['name','entity_type'])
    triples_df = pd.DataFrame(relation_triplets)

    entity_list = []
    for ent in entities:
        entity_list.append({
            'node_name': ent['text'],
            'node_id': ent['node_id'],
            'entity_type': ent['type']
        })

    node_df = pd.DataFrame(entity_list)

    return node_df, relation_df, triples_df

if __name__ == '__main__':

    text = "George Washington went to Washington"
    # text = "The three-night cruises will stop at Penang, while the four-night cruises will stop at both Penang and Port Klang near Kuala Lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital"
    text = "The three-night cruises will stop at penang, while the four-night cruises will stop at both penang and port klang near kuala lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to penang’s st george’s church and batu caves on the outskirts of the malaysian capital"
    # text = "These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital"
    # text = "He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612)."

    do_everything({'sentence': text})
