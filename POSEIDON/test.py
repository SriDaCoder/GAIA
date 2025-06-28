import requests
import os

os.system("cls" if os.name == "nt" else "clear")

def test_poseidon():
    url = "http://127.0.0.1:5000/predict"
    data = {
        "pH": 7.0,
        "turbidity": 0.0,
        "dissolved_oxygen": 100.0,
        "contaminants": 0.0,
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print("Response: Usability score:", response.json()['usability_score'], "(Expected range: 1-14)")
    else:
        print("Test failed!")
        print("Status code:", response.status_code)
        print("Response:", response.text)
test_poseidon()