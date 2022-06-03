import requests
import json
from typing import List, Dict

def predict_templates(dataset: Dict):
    response = requests.post('http://0.0.0.0:4000/ner_point', json = dataset)
    #ipdb.set_trace()
    print([doc for doc in response.iter_lines()])
    response = response.json()
    return response


if __name__ == '__main__':
    dataset = {"sentence":"Apple computers"}
    results = predict_templates(dataset)
    print(results)

