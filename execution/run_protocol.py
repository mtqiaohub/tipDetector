import json 
import time
import requests

ROBOT = "http://10.70.18.217:31950"
HEADERS = {"opentrons-version": "*"}

r = requests.get(f"{ROBOT}/protocols", headers=HEADERS, timeout=30)
r.raise_for_status()

protocols = r.json()["data"]

for p in protocols:
    print(
        p["id"],
        [f["name"] for f in p.get("files", [])],
        p.get("createdAt")
    )

print(len(protocols))


