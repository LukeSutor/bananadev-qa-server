# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {
    'question': 'Why did the chicken cross the road?',
    'text': 'The chicken crossed the road to get to the other side.'
                }

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())