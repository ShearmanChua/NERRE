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
    # dataset = {"sentence":"Apple computers"}
    dataset = {"sentence": "state of new york and the largest city in upstate new york the population was 256 304 the city is the county seat of erie county and a major gateway for commerce and travel across the canada united states border forming part of the bi national buffalo niagara region the buffalo area was inhabited before the 17th century by the native american iroquois tribe and later by french colonizers the city grew significantly in the 19th and 20th centuries as a result of immigration the construction of the erie canal and rail transportation and its close proximity to lake erie this growth provided an abundance of fresh water and an ample trade route to the midwestern united states while grooming its economy for the grain steel and automobile industries that dominated the city s economy in the 20th century since the city s economy relied heavily on manufacturing deindustrialization in the latter half of the 20th century led to a steady decline in population while some manufacturing activity remains buffalo s economy has transitioned to service industries with a greater emphasis on healthcare research and higher education which emerged following the great recession buffalo is on the eastern shore of"}
    results = predict_templates(dataset)
    print(results)

