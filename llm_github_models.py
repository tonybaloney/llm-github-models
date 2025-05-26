from typing import AsyncGenerator, Iterable, Iterator, List, Optional, Union

import llm
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    AudioContentFormat,
    AudioContentItem,
    ChatRequestMessage,
    ContentItem,
    ImageContentItem,
    ImageDetailLevel,
    ImageUrl,
    InputAudio,
    JsonSchemaFormat,
    SystemMessage,
    TextContentItem,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from llm.models import (
    AsyncConversation,
    AsyncModel,
    AsyncResponse,
    Attachment,
    Conversation,
    EmbeddingModel,
    Prompt,
    Response,
)
from pydantic import BaseModel

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

CHAT_MODELS = [
    ("AI21-Jamba-1.5-Large", True, False, ["text"], ["text"]),
    ("AI21-Jamba-1.5-Mini", True, False, ["text"], ["text"]),
    ("Codestral-2501", True, False, ["text"], ["text"]),
    ("Cohere-command-r", True, False, ["text"], ["text"]),
    ("Cohere-command-r-08-2024", True, False, ["text"], ["text"]),
    ("Cohere-command-r-plus", True, False, ["text"], ["text"]),
    ("Cohere-command-r-plus-08-2024", True, False, ["text"], ["text"]),
    ("DeepSeek-R1", True, False, ["text"], ["text"]),
    ("DeepSeek-V3", True, False, ["text"], ["text"]),
    ("DeepSeek-V3-0324", True, False, ["text"], ["text"]),
    (
        "Llama-3.2-11B-Vision-Instruct",
        True,
        False,
        ["text", "image", "audio"],
        ["text"],
    ),
    (
        "Llama-3.2-90B-Vision-Instruct",
        True,
        False,
        ["text", "image", "audio"],
        ["text"],
    ),
    ("Llama-3.3-70B-Instruct", True, False, ["text"], ["text"]),
    (
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        True,
        False,
        ["text", "image"],
        ["text"],
    ),
    ("Llama-4-Scout-17B-16E-Instruct", True, False, ["text", "image"], ["text"]),
    ("Meta-Llama-3-70B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3-8B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3.1-405B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3.1-70B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3.1-8B-Instruct", True, False, ["text"], ["text"]),
    ("Ministral-3B", True, False, ["text"], ["text"]),
    ("Mistral-Large-2411", True, False, ["text"], ["text"]),
    ("Mistral-Nemo", True, False, ["text"], ["text"]),
    ("Mistral-large", True, False, ["text"], ["text"]),
    ("Mistral-large-2407", True, False, ["text"], ["text"]),
    ("Mistral-small", True, False, ["text"], ["text"]),
    ("Phi-3-medium-128k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-medium-4k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-mini-128k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-mini-4k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-small-128k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-small-8k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3.5-MoE-instruct", True, False, ["text"], ["text"]),
    ("Phi-3.5-mini-instruct", True, False, ["text"], ["text"]),
    ("Phi-3.5-vision-instruct", True, False, ["text", "image"], None),
    ("Phi-4", True, False, ["text"], ["text"]),
    ("Phi-4-mini-instruct", True, False, ["text"], ["text"]),
    ("Phi-4-multimodal-instruct", True, False, ["audio", "image", "text"], ["text"]),
    ("gpt-4.1", True, True, ["text", "image", "audio"], ["text"]),
    ("gpt-4.1-mini", True, True, ["text", "image"], ["text"]),
    ("gpt-4.1-nano", True, True, ["text", "image"], ["text"]),
    ("gpt-4o", True, True, ["text", "image", "audio"], ["text"]),
    ("gpt-4o-mini", True, True, ["text", "image", "audio"], ["text"]),
    ("jais-30b-chat", True, False, ["text"], ["text"]),
    ("mistral-small-2503", True, False, ["text", "image"], ["text"]),
    ("o1", False, True, ["text", "image"], ["text"]),
    ("o1-mini", False, False, ["text"], ["text"]),
    ("o1-preview", False, False, ["text"], ["text"]),
    ("o3-mini", False, True, ["text"], ["text"]),
]


EMBEDDING_MODELS = [
    ("Cohere-embed-v3-english", []),
    ("Cohere-embed-v3-multilingual", []),
    ("text-embedding-3-large", [1024, 256]),
    ("text-embedding-3-small", [512]),
]


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for model_id, can_stream, supports_schema, input_modalities, output_modalities in CHAT_MODELS:
        register(
            GitHubModels(
                model_id,
                can_stream=can_stream,
                supports_schema=supports_schema,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
            ),
            GitHubAsyncModels(
                model_id,
                can_stream=can_stream,
                supports_schema=supports_schema,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
            ),
        )


@llm.hookimpl
def register_embedding_models(register):
    # Register embedding models
    for model_id, supported_dimensions in EMBEDDING_MODELS:
        register(GitHubEmbeddingModel(model_id))
        for dimensions in supported_dimensions:
            register(GitHubEmbeddingModel(model_id, dimensions=dimensions))


IMAGE_ATTACHMENTS = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

AUDIO_ATTACHMENTS = {
    "audio/wav",
    "audio/mpeg",
}


def attachment_as_content_item(attachment: Attachment) -> ContentItem:
    if attachment is None or attachment.resolve_type() is None:
        raise ValueError("Attachment cannot be None or empty")

    attachment_type: str = attachment.resolve_type()  # type: ignore

    if attachment_type.startswith("audio/"):
        audio_format = (
            AudioContentFormat.WAV if attachment_type == "audio/wav" else AudioContentFormat.MP3
        )
        if attachment.path is None:
            raise ValueError("Audio attachment must have a path for audio content")

        return AudioContentItem(
            input_audio=InputAudio.load(audio_file=attachment.path, audio_format=audio_format)
        )
    if attachment_type.startswith("image/"):
        if attachment.url:
            return ImageContentItem(
                image_url=ImageUrl(
                    url=attachment.url,
                    detail=ImageDetailLevel.AUTO,
                ),
            )
        if attachment.path:
            return ImageContentItem(
                image_url=ImageUrl.load(
                    image_file=attachment.path,
                    image_format=attachment_type.split("/")[1],
                    detail=ImageDetailLevel.AUTO,
                ),
            )

    raise ValueError(f"Unsupported attachment type: {attachment_type}")


def build_messages(
    prompt: Prompt, conversation: Optional[Union[Conversation, AsyncConversation]] = None
) -> List[ChatRequestMessage]:
    messages: List[ChatRequestMessage] = []
    current_system = None
    if conversation is not None:
        for prev_response in conversation.responses:
            if prev_response.prompt.system and prev_response.prompt.system != current_system:
                messages.append(SystemMessage(prev_response.prompt.system))
                current_system = prev_response.prompt.system
            if prev_response.attachments:
                attachment_message: list[ContentItem] = []
                if prev_response.prompt.prompt:
                    attachment_message.append(TextContentItem(text=prev_response.prompt.prompt))
                for attachment in prev_response.attachments:
                    attachment_message.append(attachment_as_content_item(attachment))
                messages.append(UserMessage(attachment_message))
            else:
                messages.append(UserMessage(prev_response.prompt.prompt))
            messages.append(AssistantMessage(prev_response.text_or_raise()))  # type: ignore
    if prompt.system and prompt.system != current_system:
        messages.append(SystemMessage(prompt.system))
    if not prompt.attachments:
        messages.append(UserMessage(content=prompt.prompt))
    else:
        attachment_message = []
        if prompt.prompt:
            attachment_message.append(TextContentItem(text=prompt.prompt))
        for attachment in prompt.attachments:
            attachment_message.append(attachment_as_content_item(attachment))
        messages.append(UserMessage(attachment_message))
    return messages


class _Shared:
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"

    def __init__(
        self,
        model_id: str,
        can_stream: bool = True,
        supports_schema: bool = False,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
    ):
        self.model_id = f"github/{model_id}"
        self.model_name = model_id
        self.can_stream = can_stream
        self.supports_schema = supports_schema
        self.attachment_types = set()
        if input_modalities and "image" in input_modalities:
            self.attachment_types.update(IMAGE_ATTACHMENTS)
        if input_modalities and "audio" in input_modalities:
            self.attachment_types.update(AUDIO_ATTACHMENTS)

        self.input_modalities = input_modalities
        self.output_modalities = output_modalities

        self.client_kwargs = {}
        self.client_kwargs["api_version"] = "2025-03-01-preview"  # Use latest version

    # Using the same display string for both the sync and async models
    # makes them not show up twice in `llm models`
    def __str__(self) -> str:
        return f"GitHub Models: {self.model_id}"


class GitHubModels(_Shared, llm.Model):
    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
        # unset keys are handled by llm.Model.get_key()
        key: str = self.get_key()  # type: ignore

        with ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
            model=self.model_name,
            **self.client_kwargs,
        ) as client:
            if prompt.schema:
                if not isinstance(prompt.schema, dict) and issubclass(prompt.schema, BaseModel):
                    response_format = JsonSchemaFormat(
                        name="output", schema=prompt.schema.model_json_schema()
                    )
                else:
                    response_format = JsonSchemaFormat(
                        name="output",
                        schema=prompt.schema,  # type: ignore[variable]
                    )
            else:
                response_format = "text"
            messages = build_messages(prompt, conversation)
            if stream:
                completion = client.complete(
                    messages=messages,
                    stream=True,
                    response_format=response_format,
                )
                chunks = []
                for chunk in completion:
                    chunks.append(chunk)
                    try:
                        content = chunk.choices[0].delta.content
                    except IndexError:
                        content = None
                    if content is not None:
                        yield content
                response.response_json = None  # TODO
            else:
                completion = client.complete(
                    messages=messages,
                    stream=False,
                    response_format=response_format,
                )
                response.response_json = None  # TODO
                yield completion.choices[0].message.content


class GitHubAsyncModels(_Shared, AsyncModel):
    async def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: AsyncResponse,
        conversation: Optional[AsyncConversation],
    ) -> AsyncGenerator[str, None]:
        key = self.get_key()

        async with AsyncChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),  # type: ignore[variable]
            model=self.model_name,
            **self.client_kwargs,
        ) as client:
            if prompt.schema:
                if not isinstance(prompt.schema, dict) and issubclass(prompt.schema, BaseModel):
                    response_format = JsonSchemaFormat(
                        name="output", schema=prompt.schema.model_json_schema()
                    )
                else:
                    response_format = JsonSchemaFormat(
                        name="output",
                        schema=prompt.schema,  # type: ignore[variable]
                    )
            else:
                response_format = "text"

            messages = build_messages(prompt, conversation)
            if stream:
                completion = await client.complete(
                    messages=messages,
                    stream=True,
                    response_format=response_format,
                )
                async for chunk in completion:
                    try:
                        content = chunk.choices[0].delta.content
                    except IndexError:
                        content = None
                    if content is not None:
                        yield content
                response.response_json = None  # TODO
            else:
                completion = await client.complete(
                    messages=messages,
                    stream=False,
                    response_format=response_format,
                )
                response.response_json = None  # TODO
                yield completion.choices[0].message.content


class GitHubEmbeddingModel(EmbeddingModel):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"
    batch_size = 100

    def __init__(self, model_id: str, dimensions: Optional[int] = None):
        self.model_id = f"github/{model_id}"
        if dimensions is not None:
            self.model_id += f"-{dimensions}"

        self.model_name = model_id
        self.dimensions = dimensions

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
        if not items:
            return iter([])

        key = self.get_key()
        client = EmbeddingsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),  # type: ignore
        )

        # TODO: Handle iterable of bytes

        kwargs = {
            "input": items,
            "model": self.model_name,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embed(**kwargs)
        return ([float(x) for x in item.embedding] for item in response.data)
