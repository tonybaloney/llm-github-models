from typing import Iterable, Iterator, List, Optional, Union

import llm
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
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
    SystemMessage,
    TextContentItem,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from llm.models import Attachment, Conversation, EmbeddingModel, Prompt, Response

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

CHAT_MODELS = [
    ("AI21-Jamba-1.5-Large", True, ["text"], ["text"]),
    ("AI21-Jamba-1.5-Mini", True, ["text"], ["text"]),
    ("Codestral-2501", True, ["text"], ["text"]),
    ("Cohere-command-r", True, ["text"], ["text"]),
    ("Cohere-command-r-08-2024", True, ["text"], ["text"]),
    ("Cohere-command-r-plus", True, ["text"], ["text"]),
    ("Cohere-command-r-plus-08-2024", True, ["text"], ["text"]),
    ("DeepSeek-R1", True, ["text"], ["text"]),
    ("DeepSeek-V3", True, ["text"], ["text"]),
    ("DeepSeek-V3-0324", True, ["text"], ["text"]),
    ("Llama-3.2-11B-Vision-Instruct", True, ["text", "image", "audio"], ["text"]),
    ("Llama-3.2-90B-Vision-Instruct", True, ["text", "image", "audio"], ["text"]),
    ("Llama-3.3-70B-Instruct", True, ["text"], ["text"]),
    ("Llama-4-Maverick-17B-128E-Instruct-FP8", True, ["text", "image"], ["text"]),
    ("Llama-4-Scout-17B-16E-Instruct", True, ["text", "image"], ["text"]),
    ("Meta-Llama-3-70B-Instruct", True, ["text"], ["text"]),
    ("Meta-Llama-3-8B-Instruct", True, ["text"], ["text"]),
    ("Meta-Llama-3.1-405B-Instruct", True, ["text"], ["text"]),
    ("Meta-Llama-3.1-70B-Instruct", True, ["text"], ["text"]),
    ("Meta-Llama-3.1-8B-Instruct", True, ["text"], ["text"]),
    ("Ministral-3B", True, ["text"], ["text"]),
    ("Mistral-Large-2411", True, ["text"], ["text"]),
    ("Mistral-Nemo", True, ["text"], ["text"]),
    ("Mistral-large", True, ["text"], ["text"]),
    ("Mistral-large-2407", True, ["text"], ["text"]),
    ("Mistral-small", True, ["text"], ["text"]),
    ("Phi-3-medium-128k-instruct", True, ["text"], ["text"]),
    ("Phi-3-medium-4k-instruct", True, ["text"], ["text"]),
    ("Phi-3-mini-128k-instruct", True, ["text"], ["text"]),
    ("Phi-3-mini-4k-instruct", True, ["text"], ["text"]),
    ("Phi-3-small-128k-instruct", True, ["text"], ["text"]),
    ("Phi-3-small-8k-instruct", True, ["text"], ["text"]),
    ("Phi-3.5-MoE-instruct", True, ["text"], ["text"]),
    ("Phi-3.5-mini-instruct", True, ["text"], ["text"]),
    ("Phi-3.5-vision-instruct", True, ["text", "image"], None),
    ("Phi-4", True, ["text"], ["text"]),
    ("Phi-4-mini-instruct", True, ["text"], ["text"]),
    ("Phi-4-multimodal-instruct", True, ["audio", "image", "text"], ["text"]),
    ("gpt-4.1", True, ["text", "image", "audio"], ["text"]),
    ("gpt-4.1-mini", True, ["text", "image"], ["text"]),
    ("gpt-4.1-nano", True, ["text", "image"], ["text"]),
    ("gpt-4o", True, ["text", "image", "audio"], ["text"]),
    ("gpt-4o-mini", True, ["text", "image", "audio"], ["text"]),
    ("jais-30b-chat", True, ["text"], ["text"]),
    ("mistral-small-2503", True, ["text", "image"], ["text"]),
    ("o1", False, ["text", "image"], ["text"]),
    ("o1-mini", False, ["text"], ["text"]),
    ("o1-preview", False, ["text"], ["text"]),
    ("o3-mini", False, ["text"], ["text"]),
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
    for model_id, can_stream, input_modalities, output_modalities in CHAT_MODELS:
        register(
            GitHubModels(
                model_id,
                can_stream=can_stream,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
            )
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
    prompt: Prompt, conversation: Optional[Conversation]
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


class GitHubModels(llm.Model):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"

    def __init__(
        self,
        model_id: str,
        can_stream: bool,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
    ):
        self.model_id = f"github/{model_id}"
        self.model_name = model_id
        self.can_stream = can_stream
        self.attachment_types = set()
        if input_modalities and "image" in input_modalities:
            self.attachment_types.update(IMAGE_ATTACHMENTS)
        if input_modalities and "audio" in input_modalities:
            self.attachment_types.update(AUDIO_ATTACHMENTS)

        self.input_modalities = input_modalities
        self.output_modalities = output_modalities

    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
        # unset keys are handled by llm.Model.get_key()
        key: str = self.get_key()  # type: ignore

        extra = {}
        if self.model_name == "o3-mini":
            extra["api_version"] = "2024-12-01-preview"

        client = ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
            model=self.model_name,
            **extra,
        )
        messages = build_messages(prompt, conversation)
        if stream:
            completion = client.complete(
                messages=messages,
                stream=True,
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
