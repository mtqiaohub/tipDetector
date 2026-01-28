import requests
import tempfile
import os

ROBOT_IP = "10.70.18.217"

HEADERS = {"opentrons-version": "3"}

#a57c644a-ce8a-4e99-ae23-98211362e7d6
#protocol id for testing http_api

protocol_id = "6f2eb7d3-3dd0-43de-95a8-7dea0846bf6e"

csv_contents = """well,has_tip
A1,1
A2,0
A3,1
B1,0
"""

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as f:
    f.write(csv_contents)
    csv_path = f.name

with open(csv_path, "rb") as f:
    r = requests.post(
        f"http://{ROBOT_IP}:31950/dataFiles",
        headers=HEADERS,
        files={"file": f},
    )
r.raise_for_status()


data_file_id = r.json()["data"]["id"]
print("Uploaded file_id:", data_file_id)


r = requests.post(
    f"http://{ROBOT_IP}:31950/runs",
    headers=HEADERS,
    json={
        "data": {
            "protocolId": protocol_id,
            "runTimeParameterFiles": {
                "tiprack_state_csv": data_file_id
            }
        }
    },
)
r.raise_for_status()

run_id = r.json()["data"]["id"]
print("Created run_id:", run_id)

# play run
r = requests.post(
    f"http://{ROBOT_IP}:31950/runs/{run_id}/actions",
    headers=HEADERS,
    json={"data": {"actionType": "play"}},
)
r.raise_for_status()

print("Run started")

os.unlink(csv_path)
