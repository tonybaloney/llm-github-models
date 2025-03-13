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
        if model["task"] == "chat-completion":
            chat_models.append(
                (
                    model["original_name"],
                    supports_streaming(model["name"]),
                    model["supported_input_modalities"],
                    model["supported_output_modalities"],
                )
            )
        elif model["task"] == "embeddings":
            embedding_models.append(model["original_name"])
        else:
            print("Not sure what to do with this model: ", model["name"])

print("Chat models:")
# sort by name
chat_models = sorted(chat_models, key=lambda x: x[0])
pprint(chat_models)
print("Embedding models:")
# sort by name
embedding_models = sorted(embedding_models)
pprint(embedding_models)

## Make a Markdown series for the models

with open("models.fragment.md", "w", encoding="utf-8") as f:
    f.write("## Supported Models\n\n")

    for model in models:
        f.write(f"### {model['friendly_name']}\n\n")
        f.write(f"![Model Image](https://github.com/{model['logo_url']})\n\n")
        f.write(f"Usage: `llm -m github/{model['name']}`\n\n")
        f.write(f"**Publisher:** {model["publisher"]} \n\n")
        f.write(f"**Description:** {model["description"].replace("\n## ", "\n#### ")} \n\n")
