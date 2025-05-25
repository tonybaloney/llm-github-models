"""
A script to parse the models.json from the github API until there is a live API to call.
"""

import json
from pprint import pprint

chat_models = []
embedding_models = []


def supports_streaming(name):
    if name in ["o1", "o1-mini", "o1-preview", "o3-mini"]:
        return False
    return True


with open("models.json", "r", encoding="utf-8") as f:
    models = json.load(f)
    for model in models:
        if "chat-completion" in model["inferenceTasks"]:
            chat_models.append(
                (
                    model["name"],
                    supports_streaming(model["name"]),
                    model["modelLimits"]["supportedInputModalities"],
                    model["modelLimits"]["supportedOutputModalities"],
                )
            )
        elif "embeddings" in model["inferenceTasks"]:
            embedding_models.append(model["name"])
        else:
            print("Not sure what to do with this model: ", model["name"])

print("Chat models:")
# sort by name
chat_models = sorted(chat_models, key=lambda x: x[0])
pprint(chat_models, indent=4, width=999)
print("Embedding models:")
# sort by name
embedding_models = sorted(embedding_models)
pprint(embedding_models, indent=4)

## Make a Markdown series for the models

with open("models.fragment.md", "w", encoding="utf-8") as f:
    f.write("## Supported Models\n\n")

    for model in models:
        f.write(f"### {model['displayName']}\n\n")
        f.write(f"Usage: `llm -m github/{model['name']}`\n\n")
        f.write(f"**Publisher:** {model["publisher"]} \n\n")
        f.write(f"**Description:** {model["summary"].replace("\n## ", "\n#### ")} \n\n")
