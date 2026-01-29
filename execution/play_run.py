import requests
import tempfile
import os
import yaml

with open('http_config.yaml', 'r') as file:
    configuration = yaml.safe_load(file)

ROBOT_IP = configuration['ROBOT_IP']
HEADERS = {"opentrons-version": configuration['VERSION']}

protocol_id = configuration["PROTOCOL_ID"]

# csv_contents = """well,has_tip
# A1,1
# A2,0
# A3,1
# B1,0
# """
# with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as f:
#     f.write(csv_contents)
#     csv_path = f.name

def create_run(protocol_id):

    with open(csv_path, "rb") as f:
        r = requests.post(
            f"http://{ROBOT_IP}:31950/dataFiles",
            headers=HEADERS,
            files={"file": f},
        )
        
    r.raise_for_status()

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

    data_file_id = r.json()["data"]["id"]
    print("Uploaded file_id:", data_file_id)

    run_id = r.json()["data"]["id"]
    print("Created run_id:", run_id)


def execute_run(run_id):
    r = requests.post(
        f"http://{ROBOT_IP}:31950/runs/{run_id}/actions",
        headers=HEADERS,
        json={"data": {"actionType": "play"}},
    )
    r.raise_for_status()

    print("Run started")

# os.unlink(csv_path)
