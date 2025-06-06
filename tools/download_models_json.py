import json

import requests

url = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"
headers = {"Content-Type": "application/json"}
filters = {
    "filters": [
        {"field": "freePlayground", "operator": "eq", "values": ["true"]},
        {"field": "labels", "operator": "eq", "values": ["latest"]},
    ],
    "order": [{"field": "name", "direction": "asc"}],
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

gtihub_models_url = "https://models.github.ai/catalog/models"
headers = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

print("Fetching GitHub models...")
response = requests.get(gtihub_models_url, headers=headers)
response.raise_for_status()

github_models = response.json()
github_models = set(model["name"] for model in github_models)

models = filter(lambda model: model["displayName"] in github_models, all_models)
models = sorted(models, key=lambda x: x["name"])

print("Saving models to models.json...")
with open("models.json", "w") as f:
    json.dump(models, f, indent=4)

print(f"Saved {len(models)} models to models.json")
