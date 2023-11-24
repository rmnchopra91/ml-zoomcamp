import requests

# url = 'http://localhost:8082/2015-03-31/functions/function/invocations'
url = 'http://localhost:8082/'

data = {'url': 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'}

result = requests.post(url, json=data).json()
print(result)