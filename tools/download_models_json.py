import requests
import json

url = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"
headers = {
    "Content-Type": "application/json"
}
filters = {
    "filters": [
        {
            "field": "freePlayground",
            "operator": "eq",
            "values": ["true"]
        },
        {
            "field": "labels",
            "operator": "eq",
            "values": ["latest"]
        }
    ],
    "order": [
        {
            "field": "name",
            "direction": "asc"
        }
    ]
}

all_models = []
continuation_token = None

while True:
    payload = filters.copy()
    if continuation_token:
        payload["continuationToken"] = continuation_token

    print("Fetching models...")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    all_models.extend(data.get("summaries", []))

    continuation_token = data.get("continuationToken")
    if continuation_token:
        print(f"Continuation token: {continuation_token}")
    if not continuation_token:
        break

print("Saving models to models.json...")
with open("models.json", "w") as f:
    json.dump(all_models, f, indent=4)

print(f"Saved {len(all_models)} models to models.json")
