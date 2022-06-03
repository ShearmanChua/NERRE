import opennre
from flair.data import Sentence
from flair.models import SequenceTagger
import time

# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")

model = opennre.get_model('wiki80_bert_softmax').cuda()

start = time.time()

text = "George Washington went to Washington"
# text = "The three-night cruises will stop at Penang, while the four-night cruises will stop at both Penang and Port Klang near Kuala Lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital"
text = "The three-night cruises will stop at penang, while the four-night cruises will stop at both penang and port klang near kuala lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to penang’s st george’s church and batu caves on the outskirts of the malaysian capital"
# text = "These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital"
# text = "He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612)."

# make Flair sentence
sentence = Sentence(text)
# predict NER tags
tagger.predict(sentence)

entities = []
# print predicted NER spans
# print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    # print(entity)
    ent = {}
    ent['start_pos'] = entity.tokens[0].start_pos
    ent['end_pos'] = entity.tokens[-1].end_pos
    ent['text'] = text[entity.tokens[0].start_pos:entity.tokens[-1].end_pos]
    entities.append(ent)

head_tail_pairs = []

for head_idx in range(0,len(entities)):
    head_span = (entities[head_idx]['start_pos'],entities[head_idx]['end_pos'])
    head_text = entities[head_idx]['text']
    for tail_idx in range(head_idx+1,len(entities)):
        tail_span = (entities[tail_idx]['start_pos'],entities[tail_idx]['end_pos'])
        tail_text = entities[tail_idx]['text']
        if head_text != tail_text:
            head_tail_pairs.append({'head_span':head_span,'tail_span':tail_span,'head_text':head_text,'tail_text':tail_text})

print(head_tail_pairs)
end1 = time.time()
print('time taken for pair population:', end1-start)

for pair in head_tail_pairs:
    results = model.infer({'text': text, 'h': {'pos': pair['head_span']}, 't': {'pos': pair['tail_span']}})
    # if results[1] > 0.7:
        # print(results)
        # print(pair['head_text'],results[0],pair['tail_text'])

end = time.time()
print('time taken for RE', end-start)
