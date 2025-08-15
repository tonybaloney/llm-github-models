import json

import requests

url = "https://models.github.ai/catalog/models"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

print("Fetching models...")
response = requests.get(url, headers=headers, timeout=300)
response.raise_for_status()

all_models = response.json()

print("Saving models to models.json...")
with open("models.json", "w") as f:
    json.dump(all_models, f, indent=4)

print(f"Saved {len(all_models)} models to models.json")
