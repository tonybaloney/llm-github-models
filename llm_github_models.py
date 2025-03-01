import llm
from llm.models import Attachment, Conversation, Prompt, Response
from typing import Optional, Iterator, List, Dict, Any, Tuple
from pathlib import Path
import json
import time
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import (
    ChatRequestMessage,
    AssistantMessage,
    AudioContentItem,
    TextContentItem,
    ImageContentItem,
    ContentItem,
    InputAudio,
    AudioContentFormat,
    ImageDetailLevel,
    ImageUrl,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel
import requests

# Using the updated endpoint URL
INFERENCE_ENDPOINT = "https://models.github.ai/inference"

EMBEDDING_MODELS = [
    "Cohere-embed-v3-english",
    "Cohere-embed-v3-multilingual", 
    "text-embedding-3-large",
    "text-embedding-3-small",
]


@llm.hookimpl
def register_models(register):
    """Register both sync and async versions of each model"""
    # Dynamically fetch the list of models
    models = fetch_models()    
    for model_id, can_stream, input_modalities, output_modalities in models:
        register(
            GitHubModels(
                model_id,
                can_stream=can_stream,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
            )
        )


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

class GitHubModelsConfig(BaseModel):
    GITHUB_MARKETPLACE_BASE_URL: str = "https://github.com/marketplace"

def get_cached_models(path: Path, cache_timeout: int = 3600) -> Optional[List[Tuple[str, bool, List[str], List[str]]]]:
    """
    Get cached models if they exist and aren't expired.
    
    Args:
        path: Path to cache file
        cache_timeout: Cache timeout in seconds (default: 1 hour)
        
    Returns:
        List of model tuples if valid cache exists, None otherwise
    """
    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            try:
                with open(path, "r") as file:
                    return json.load(file)
            except Exception:
                return None
    return None

def save_models_cache(models: List[Tuple[str, bool, List[str], List[str]]], path: Path):
    """
    Save models to cache file.
    
    Args:
        models: List of model tuples to cache
        path: Path to cache file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(models, file)

def fetch_models() -> List[Tuple[str, bool, List[str], List[str]]]:
    """Fetch available chat models from GitHub Marketplace with caching"""
    config = GitHubModelsConfig()
    cache_path = llm.user_dir() / "github_models.json"
    
    # Check cache first
    cached_models = get_cached_models(cache_path)
    if cached_models:
        return cached_models
    
    # Fallback models in case API fetch fails
    fallback_models = [
        ("Meta-Llama-3-70B-Instruct", True, ["text"], ["text"]),
        ("o3-mini", False, ["text"], ["text"])
    ]
    
    try:
        headers = {"Accept": "application/json"}
        models = []
        page = 1
        
        while True:
            response = requests.get(
                f"{config.GITHUB_MARKETPLACE_BASE_URL}?page={page}&type=models&task=chat-completion&query=sort%3Aname-asc",
                headers=headers,
                timeout=10
            )
            if not response.ok:
                return fallback_models
                
            data = response.json()
            
            for model in data.get("results", []):
                model_id = model.get("original_name", model["name"])
                
                # Determine capabilities based on model metadata
                input_modalities = ["text"]
                if model.get("vision_enabled"):
                    input_modalities.append("image")
                if model.get("audio_enabled"):
                    input_modalities.append("audio")
                    
                output_modalities = ["text"]
                
                # Special handling for o1 models which don't support streaming
                can_stream = not model_id.startswith("o1")
                
                models.append((
                    model_id,
                    can_stream,
                    input_modalities,
                    output_modalities
                ))

            if page >= data.get("totalPages", 1):
                break
            page += 1

        # Cache the results before returning
        if models:
            save_models_cache(models, cache_path)
            return models
        return fallback_models

    except Exception as e:
        print(f"Error fetching models, using fallback list: {e}")
        return fallback_models

def attachment_as_content_item(attachment: Attachment) -> ContentItem:
    if attachment.resolve_type().startswith("audio/"):
        audio_format = (
            AudioContentFormat.WAV
            if attachment.resolve_type() == "audio/wav"
            else AudioContentFormat.MP3
        )
        return AudioContentItem(
            input_audio=InputAudio.load(
                audio_file=attachment.path, audio_format=audio_format
            )
        )
    if attachment.resolve_type().startswith("image/"):
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
                    image_format=attachment.resolve_type().split("/")[1],
                    detail=ImageDetailLevel.AUTO,
                ),
            )

    raise ValueError(f"Unsupported attachment type: {attachment.resolve_type()}")


def build_messages(
    prompt: Prompt, conversation: Optional[Conversation]
) -> List[ChatRequestMessage]:
    messages: List[ChatRequestMessage] = []
    current_system = None
    if conversation is not None:
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(SystemMessage(prev_response.prompt.system))
                current_system = prev_response.prompt.system
            if prev_response.attachments:
                attachment_message: list[ContentItem] = []
                if prev_response.prompt.prompt:
                    attachment_message.append(
                        TextContentItem(text=prev_response.prompt.prompt)
                    )
                for attachment in prev_response.attachments:
                    attachment_message.append(attachment_as_content_item(attachment))
                messages.append(UserMessage(attachment_message))
            else:
                messages.append(UserMessage(prev_response.prompt.prompt))
            messages.append(AssistantMessage(prev_response.text_or_raise()))
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
        key = self.get_key()

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
