"""
A script to parse the models.json from the github API until there is a live API to call.
"""

import json
from pprint import pprint

chat_models = []
embedding_models = []


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
    # Note: this list does not line up with the official docs at
    # https://learn.microsoft.com/en-us/azure/machine-learning/concept-models-featured?view=azureml-api-2
    # But in practice these are the models that work.
    tool_supporting_models = [
        "o3",
        "o3-mini",
        "o4-mini",
        "o1",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "grok-3",
        "grok-3-mini",
        "cohere-command-a",
        "Cohere-command-r-plus-08-2024",
        "Cohere-command-r-08-2024",
        "Cohere-command-r-plus",
        "Cohere-command-r",
        "Codestral-2501",
        "Ministral-3B",
        "Mistral-Nemo",
        "Mistral-Large-2411",
        "Mistral-large-2407",
        "Mistral-large",
        "mistral-medium-2505",
        "mistral-small-2503",
        "Mistral-small",
    ]
    return name in tool_supporting_models


def extra_embedding_dimensions(name):
    if name == "text-embedding-3-large":
        return [1024, 256]
    elif name == "text-embedding-3-small":
        return [512]
    elif name == "embed-v-4-0":
        return [256, 512, 1024]

    return []


with open("models.json", "r", encoding="utf-8") as f:
    models = json.load(f)
    for model in models:
        if "chat-completion" in model["inferenceTasks"]:
            chat_models.append(
                (
                    model["name"],
                    supports_schemas(model["name"]),
                    requires_usage_stream_option(model["name"]),
                    supports_tools(model["name"]),
                    model["modelLimits"]["supportedInputModalities"],
                    model["modelLimits"]["supportedOutputModalities"],
                )
            )
        elif "embeddings" in model["inferenceTasks"]:
            embedding_models.append((model["name"], extra_embedding_dimensions(model["name"])))
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

    # Add chat models table
    f.write("### Chat Models\n\n")
    f.write("| Model Name | Schemas | Tools | Input Modalities | Output Modalities |\n")
    f.write("|------------|---------|-------|------------------|-------------------|\n")

    for (
        model_name,
        schemas,
        usage_stream,
        tools,
        input_modalities,
        output_modalities,
    ) in chat_models:
        schemas_str = "✅" if schemas else "❌"
        tools_str = "✅" if tools else "❌"
        input_str = ", ".join(input_modalities) if input_modalities else "text"
        output_str = ", ".join(output_modalities) if output_modalities else "text"

        f.write(f"| {model_name} | {schemas_str} | {tools_str} | {input_str} | {output_str} |\n")

    f.write("\n")

    for model in models:
        f.write(f"### {model['displayName']}\n\n")
        f.write(f"Usage: `llm -m github/{model['name']}`\n\n")
        f.write(f"**Publisher:** {model['publisher']} \n\n")
        f.write(f"**Description:** {model['summary'].replace('\n## ', '\n#### ')} \n\n")
