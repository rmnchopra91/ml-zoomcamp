import requests

api_url = "http://localhost:9696/"

def test_logistic_regression():
    input_data = {'age': 39.0,
                  'job': 'housemaid',
                  'marital': 'married',
                  'education': 'tertiary',
                  'balance': 315.0,
                  'housing': 'no',
                  'loan': 'no',
                  'contact': 'cellular',
                  'day': 28,
                  'month': 'aug',
                  'duration': 53,
                  'campaign': 4,
                  'pdays': -1,
                  'previous': 0,
                  'poutcome': 'unknown'
                }
    predict_api = api_url + "predict"
    print(f"predict api calling : {predict_api}")

    response = requests.post(predict_api, json=input_data).json()
    print(f"Here is our prediction is {response}")

test_logistic_regression()