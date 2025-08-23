import requests

url = "http://sridacoder.github.io/GAIA/GAIA-main/infer"

payload = {
    "password": "MySecretPassword123",
    "soil": [0.1, -0.2, 0.0, 0.3, 0.1, 0.0, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0, 0.1],
    "water": [0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0, 0.1, -0.2, 0.0, 0.1, 0.0],
    "weather": [0.1, 0.0, -0.1, 0.2, 0.0, 0.1, -0.2, 0.0]
}

response = requests.post(url, json=payload)
print(response.json())