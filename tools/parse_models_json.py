"""
A script to parse the models.json from the github API until there is a live API to call.
"""

import json
from pprint import pprint

chat_models = []
embedding_models = []


with open("models.json", "r") as f:
    models = json.load(f)
    for model in models:
        if model['task'] == 'chat-completion':
            chat_models.append((model['original_name'], "o1" in model['name'], model['supported_input_modalities'], model['supported_output_modalities']))
        elif model['task'] == 'embeddings':
            embedding_models.append(model['original_name'])
        else:
            print("Not sure what to do with this model: ", model['name'])

print("Chat models:")
# sort by name
chat_models = sorted(chat_models, key=lambda x: x[0])
pprint(chat_models)
print("Embedding models:")
# sort by name
embedding_models = sorted(embedding_models)
pprint(embedding_models)

## Make a Markdown series for the models

print("## Supported Models")

for model in models:
    print(f"### {model['friendly_name']}")
    print(f"![Model Image](https://github.com/{model['logo_url']})")
    print(f"Usage: `llm -m github/{model['name']}`\n")
    print("**Publisher:** ", model['publisher'], "\n")
    print("**Description:** ", model['description'].replace("\n## ", "\n#### "), "\n")

