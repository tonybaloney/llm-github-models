"""
A script to parse the models.json from the github API until there is a live API to call.
"""

import json

chat_models = []
embedding_models = []


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
                    "agents" in model["capabilities"],
                    "streaming" in model["capabilities"],
                    "tool-calling" in model["capabilities"],
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
chat_models = sorted(chat_models, key=lambda x: x[0])
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
print(
    ",\n".join(
        [
            f"EmbeddingModelSpec(llm_id='{model[0]}', github_id='{model[1]}', name='{model[2]}', dimensions={model[3]})"  # noqa: E501
            for model in embedding_models
        ]
    )
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
