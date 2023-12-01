import requests
from time import sleep

url = "http://localhost:9696/predict"
client = {"job": "retired", "duration": 445, "poutcome": "success"}
print("request ready for process")
response = requests.post(url, json=client).json()
print(f"{response}")

while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)