import requests

url = "http://127.0.0.1:9697/predict"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
response = requests.post(url, json=client).json()
print(f"-------------------------{url}----------------------------------------------")
print(response)
print("///////////////////////////////////////////////////////////////////////")
