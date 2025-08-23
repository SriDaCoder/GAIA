"""
This code:
 - Parses microcontroller CSV lines reliably.

 - Checks correct number of values (16 soil, 16 water, 8 weather).

 - Batches multiple readings if needed.

 - Retries API send up to 3 times with delay.

 - Handles exceptions without crashing.
"""

import requests
import serial
import time

API_URL = "http://<your-server-ip>:5000/infer"
API_PASSWORD = "MySecretPassword123"
SERIAL_PORT = "COM3"       # Replace with your port
BAUD_RATE = 115200
RETRY_DELAY = 5            # seconds
BATCH_SIZE = 1             # send every line; increase if needed

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

def parse_line(line: str):
    """
    Parses a line like:
    soil1,...,soil16;water1,...,water16;weather1,...,weather8
    Returns (soil:list, water:list, weather:list)
    """
    try:
        soil_str, water_str, weather_str = line.strip().split(";")
        soil = [float(x) for x in soil_str.split(",")]
        water = [float(x) for x in water_str.split(",")]
        weather = [float(x) for x in weather_str.split(",")]
        if len(soil)!=16 or len(water)!=16 or len(weather)!=8:
            raise ValueError("Incorrect number of values")
        return soil, water, weather
    except Exception as e:
        print("Parse error:", e)
        return None

def send_to_api(soil, water, weather):
    payload = {
        "password": API_PASSWORD,
        "soil": soil,
        "water": water,
        "weather": weather
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            return r.json()
        except Exception as e:
            print(f"Send attempt {attempt+1} failed:", e)
            time.sleep(RETRY_DELAY)
    return None

def main_loop():
    batch = []
    while True:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue
            vals = parse_line(line)
            if vals:
                batch.append(vals)
            if len(batch) >= BATCH_SIZE:
                for soil, water, weather in batch:
                    resp = send_to_api(soil, water, weather)
                    if resp:
                        print(resp)
                batch.clear()
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print("Unexpected error:", e)

if __name__=="__main__":
    main_loop()
