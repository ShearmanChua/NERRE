import time
import requests
from typing import List, Dict

import torch
import flair
import opennre
from flair.data import Sentence
from flair.models import SequenceTagger

text = "George Washington went to Washington"
# text = "The three-night cruises will stop at Penang, while the four-night cruises will stop at both Penang and Port Klang near Kuala Lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital"
text = "The three-night cruises will stop at penang, while the four-night cruises will stop at both penang and port klang near kuala lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to penang’s st george’s church and batu caves on the outskirts of the malaysian capital"
# text = "These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital"
# text = "He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612)."

flair.device = torch.device('cpu')

def load_tagger(name: str) -> SequenceTagger:
    return SequenceTagger.load(name)

def load_re_model(name: str, device: str = 'cuda') -> opennre.model.SoftmaxNN:
    model = opennre.get_model(name)
    if device == 'cuda':
        model = model.cuda()
    return model

def predict_templates(dataset: List[Dict]):
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

        entities.append(ent)
        el_temps.append({
            'left_context': text[:start_pos],
            'mention': ent['text'],
            'right_context': text[end_pos:]
        })

    return entities, el_temps

# load tagger
tagger = load_tagger("flair/ner-english-large")
model = load_re_model('wiki80_bert_softmax', device='cpu')

start1 = time.time()

entities, el_templates = generate_named_entities(tagger, text)

end1 = time.time()
print('Time taken for NER:', end1-start1)

top_candidate_list, resolved_entities = predict_templates(el_templates)

end2 = time.time()
print('Time taken for EL:', end2-end1)
print(resolved_entities)

filtered_entities = [ent for cand, ent in zip(top_candidate_list, entities) if cand is not None]
print(filtered_entities)

head_tail_pairs = []

for head_idx in range(0,len(filtered_entities)):
    head_span = (filtered_entities[head_idx]['start_pos'],filtered_entities[head_idx]['end_pos'])
    head_text = filtered_entities[head_idx]['text']
    for tail_idx in range(head_idx+1,len(filtered_entities)):
        tail_span = (filtered_entities[tail_idx]['start_pos'],filtered_entities[tail_idx]['end_pos'])
        tail_text = filtered_entities[tail_idx]['text']
        if head_text != tail_text:
            head_tail_pairs.append({'head_span':head_span,'tail_span':tail_span,'head_text':head_text,'tail_text':tail_text})

print(head_tail_pairs)
end1 = time.time()
print('time taken for pair population:', end1-start)

for pair in head_tail_pairs:
    results = model.infer({'text': text, 'h': {'pos': pair['head_span']}, 't': {'pos': pair['tail_span']}})
    if results[1] > 0.7:
        print(results)
        print(pair['head_text'],results[0],pair['tail_text'])

end = time.time()
print('time taken for RE', end-start)
