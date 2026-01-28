import requests
import json


ON = json.dumps({"on": True})
OFF = json.dumps({"on": False})

# Toggle for lights on/off
lights_status = OFF

ROBOT_IP = "10.70.18.217"

HEADERS = {"opentrons-version": "*"}

lights_url = f"http://{ROBOT_IP}:31950/robot/lights"

r = requests.post(
	url=lights_url,
	headers=HEADERS,
	data=lights_status)

print(f"Request status:\n{r}\n{r.text}")

r = requests.get(
	url=lights_url,
	headers=HEADERS
	)

print(f"Request status:\n{r}\n{r.text}")