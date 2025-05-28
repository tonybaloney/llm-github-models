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


def supports_schemas(name):
    if name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "o3-mini",
    ]:
        return True
    return False


def requires_usage_stream_option(name):
    return name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o4-mini",
    ]


def supports_tools(name):
    tool_supporting_models = [
        "AI21-Jamba-1.5-Mini",
        "AI21-Jamba-1.5-Large",
        "o3-mini",
        "o1",
        "o1-preview",
        "o1-mini",
        "gpt-4o-realtime-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "cohere-command-a",
        "Cohere-command-a",
        "Cohere-command-r-plus-08-2024",
        "Cohere-command-r-08-2024",
        "Cohere-command-r-plus",
        "Cohere-command-r",
        "jais-30b-chat",
        "DeepSeek-V3-0324",
        "DeepSeek-V3",
        "Llama-4-Scout-17B-16E-Instruct",
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Mistral-Large-2411",
        "Mistral-large-2407",
        "Mistral-large",
        "Mistral-small-2503",
        "Mistral-small",
    ]
    return name in tool_supporting_models


with open("models.json", "r", encoding="utf-8") as f:
    models = json.load(f)
    for model in models:
        if "chat-completion" in model["inferenceTasks"]:
            chat_models.append(
                (
                    model["name"],
                    supports_streaming(model["name"]),
                    supports_schemas(model["name"]),
                    requires_usage_stream_option(model["name"]),
                    supports_tools(model["name"]),
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

# Make a Markdown series for the models

with open("models.fragment.md", "w", encoding="utf-8") as f:
    f.write("## Supported Models\n\n")

    for model in models:
        f.write(f"### {model['displayName']}\n\n")
        f.write(f"Usage: `llm -m github/{model['name']}`\n\n")
        f.write(f"**Publisher:** {model['publisher']} \n\n")
        f.write(f"**Description:** {model['summary'].replace('\n## ', '\n#### ')} \n\n")
