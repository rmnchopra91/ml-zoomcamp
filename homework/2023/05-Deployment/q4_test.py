import requests

url = "http://localhost:9696/predict"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
print("request ready for process")
response = requests.post(url, json=client).json()
print(f"{response}")