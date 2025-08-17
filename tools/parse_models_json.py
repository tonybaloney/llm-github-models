"""
A script to parse the models.json from the github API until there is a live API to call.
"""

import json

chat_models = []
embedding_models = []


def supports_schemas(name):
    if name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # "gpt-5-chat", Leaving this here as a note to future self. It does not work.
        "o1",
        "o1-mini",  # does not work
        "o3-mini",
        "o4-mini",
        "grok-3",
        "grok-3-mini",
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
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-chat",
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
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-chat",
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
        id = model["id"].split("/")[-1]
        if "text" in model["supported_output_modalities"]:
            chat_models.append(
                (
                    id,
                    model["id"],
                    model["name"],
                    supports_schemas(id),
                    requires_usage_stream_option(id),
                    "tool-calling" in model["capabilities"] or supports_tools(id),
                    model["supported_input_modalities"],
                    model["supported_output_modalities"],
                )
            )
        elif "embeddings" in model["supported_output_modalities"]:
            embedding_models.append(
                (id, model["id"], model["name"], extra_embedding_dimensions(id))
            )
        else:
            print(
                "Not sure what to do with this model: ",
                model["name"],
                model["supported_output_modalities"],
            )

print("Chat models:")
# sort by name
chat_models = sorted(chat_models, key=lambda x: x[1])
print("[")
print(
    ",\n".join(
        [
            f"ChatModelSpec(llm_id='{model[0]}', github_id='{model[1]}', name='{model[2]}', supports_schemas={model[3]}, supports_streaming={model[4]}, supports_tools={model[5]}, supported_input_modalities={model[6]}, supported_output_modalities={model[7]})"  # noqa: E501
            for model in chat_models
        ]
    )
)
print("]\n\n")
print("Embedding models:")
# sort by name
embedding_models = sorted(embedding_models)

print("[")

for model in embedding_models:
    if not model[3]:
        print(
            f"EmbeddingModelSpec(llm_id='{model[0]}', github_id='{model[1]}', name='{model[2]}', dimensions=None),"
        )
    else:
        for dim in model[3]:
            print(
                f"EmbeddingModelSpec(llm_id='{model[0]}-{dim}', github_id='{model[1]}', name='{model[2]} ({dim})', dimensions={dim}),"
            )
print("]\n\n")

with open("models.fragment.md", "w", encoding="utf-8") as f:
    f.write("## Supported Models\n\n")

    # Add chat models table
    f.write("### Chat Models\n\n")
    f.write("| Model Name | Schemas | Tools | Input Modalities | Output Modalities |\n")
    f.write("|------------|---------|-------|------------------|-------------------|\n")

    for (
        _,
        _,
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
        f.write(f"### {model['name']}\n\n")
        f.write(f"Usage: `llm -m github/{model['id'].split('/')[-1]}`\n\n")
        f.write(f"**Publisher:** {model['publisher']} \n\n")
        f.write(f"**Description:** {model['summary'].replace('\n## ', '\n#### ')} \n\n")
