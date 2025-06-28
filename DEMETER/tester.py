import requests
import os, time

os.system("cls" if os.name == "nt" else "clear")

def test_demeter():
    url = "http://127.0.0.1:5000/predict"
    data = {
        "sunlight": 10.0,
        "water": 50.0,
        "soil_quality": 10.0,
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        for i in str("Response: Usability score: " + str(response.json()['usability_score']) + " (Expected range: 1-10)"):
            print(i, end="")
            time.sleep(0.1)
        print()
    else:
        for i in str("Test failed! Status code: " + str(response.status_code) + "; Response received: " + str(response.text)):
            print(i, end="")
            time.sleep(0.1)
test_demeter()