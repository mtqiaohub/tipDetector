import requests
import json


ROBOT_IP = "10.70.18.217"
RUN_ID = "4939d899-1ec0-4aff-bffc-f0d1a96ba1f8"

HEADERS = {"opentrons-version": "3"}

#4939d899-1ec0-4aff-bffc-f0d1a96ba1f8 ['test_http_api.py', 'eppendorf_rack_lid.json', 'custom_tilting_reservoir_7500_tilted.json', 'custom_tilting_reservoir_lower.json', 'custom_tilting_reservoir_7500.json', 'biotdot96wellpcrplate_96_wellplate_200ul.json'] 2026-01-27T22:16:30.216298Z
#protocol id for testing http_api

protocol_id = "4939d899-1ec0-4aff-bffc-f0d1a96ba1f8"

# create run
r = requests.post(
    f"http://{ROBOT_IP}:31950/runs",
    headers=HEADERS,
    json={"data": {"protocolId": protocol_id}},
)
r.raise_for_status()
run_id = r.json()["data"]["id"]

# play run
r = requests.post(
    f"http://{ROBOT_IP}:31950/runs/{run_id}/actions",
    headers=HEADERS,
    json={"data": {"actionType": "play"}},
)
r.raise_for_status()

