import json
import os
from typing import AsyncGenerator, Dict, Iterable, Iterator, List, Optional, Union

import llm
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    AudioContentFormat,
    AudioContentItem,
    ChatCompletionsToolCall,
    ChatCompletionsToolDefinition,
    ChatRequestMessage,
    CompletionsUsage,
    ContentItem,
    FunctionCall,
    FunctionDefinition,
    ImageContentItem,
    ImageDetailLevel,
    ImageUrl,
    InputAudio,
    JsonSchemaFormat,
    StreamingChatResponseMessageUpdate,
    StreamingChatResponseToolCallUpdate,
    SystemMessage,
    TextContentItem,
    ToolMessage,
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
    Model,
    Prompt,
    Response,
    ToolCall,
)
from pydantic import BaseModel

from llm_github_models.chat_models import CHAT_MODELS
from llm_github_models.embedding_models import EMBEDDING_MODELS

INFERENCE_ENDPOINT = "https://models.github.ai/inference"


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for spec in CHAT_MODELS:
        register(
            GitHubModels(
                model_id=spec.llm_id,
                github_id=spec.github_id,
                model_name=spec.name,
                supports_schema=spec.supports_schemas,
                requires_usage_stream_option=spec.supports_streaming,
                supports_tools=spec.supports_tools,
                input_modalities=spec.supported_input_modalities,
                output_modalities=spec.supported_output_modalities,
            ),
            GitHubAsyncModels(
                model_id=spec.llm_id,
                github_id=spec.github_id,
                model_name=spec.name,
                supports_schema=spec.supports_schemas,
                requires_usage_stream_option=spec.supports_streaming,
                supports_tools=spec.supports_tools,
                input_modalities=spec.supported_input_modalities,
                output_modalities=spec.supported_output_modalities,
            ),
        )


@llm.hookimpl
def register_embedding_models(register):
    # Register embedding models
    for model in EMBEDDING_MODELS:
        register(GitHubEmbeddingModel(model.llm_id, model.github_id, model.name, model.dimensions))


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
            elif prev_response.prompt.prompt:
                messages.append(UserMessage(prev_response.prompt.prompt))

            # Add any tool results from the previous prompt
            for tool_result in prev_response.prompt.tool_results:
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_result.tool_call_id or "", content=tool_result.output
                    )
                )

            # Add the assistant's response
            assistant_msg = AssistantMessage(prev_response.text_or_raise())  # type: ignore

            tool_calls = prev_response.tool_calls_or_raise()  # type: ignore
            if tool_calls:
                assistant_tool_calls = []
                for tool_call in tool_calls:
                    assistant_tool_calls.append(
                        ChatCompletionsToolCall(
                            id=tool_call.tool_call_id,
                            function=FunctionCall(
                                name=tool_call.name, arguments=json.dumps(tool_call.arguments)
                            ),
                        )
                    )

                # Set tool_calls on the assistant message
                assistant_msg.tool_calls = assistant_tool_calls

            messages.append(assistant_msg)

    if prompt.system and prompt.system != current_system:
        messages.append(SystemMessage(prompt.system))
    if prompt.attachments:
        attachment_message = []
        if prompt.prompt:
            attachment_message.append(TextContentItem(text=prompt.prompt))
        for attachment in prompt.attachments:
            attachment_message.append(attachment_as_content_item(attachment))
        messages.append(UserMessage(attachment_message))
    elif prompt.prompt:
        messages.append(UserMessage(content=prompt.prompt))

    # Add any tool results for the current prompt
    for tool_result in prompt.tool_results:
        messages.append(
            ToolMessage(tool_call_id=tool_result.tool_call_id or "", content=tool_result.output)
        )

    return messages


def set_usage(usage: CompletionsUsage, response: Union[Response, AsyncResponse]) -> None:
    # Recursively remove keys with value 0 and empty dictionaries
    def remove_empty_and_zero(obj):
        if isinstance(obj, dict):
            cleaned = {k: remove_empty_and_zero(v) for k, v in obj.items() if v != 0 and v != {}}
            return {k: v for k, v in cleaned.items() if v is not None and v != {}}
        return obj

    details = usage.as_dict()
    details.pop("prompt_tokens", None)
    details.pop("completion_tokens", None)
    details.pop("total_tokens", None)

    response.set_usage(
        input=usage.prompt_tokens,
        output=usage.completion_tokens,
        details=remove_empty_and_zero(details),
    )


def append_streaming_tool_calls(
    tool_calls: Dict[str, StreamingChatResponseToolCallUpdate],
    delta: StreamingChatResponseMessageUpdate,
):
    if not delta.tool_calls:
        return

    for tool_call in delta.tool_calls:
        index = tool_call.get("index")
        if index not in tool_calls:
            tool_calls[index] = tool_call
        else:
            tool_calls[index].function.arguments += tool_call.function.arguments


def add_tool_calls(
    tool_calls: Iterable[Union[ChatCompletionsToolCall, StreamingChatResponseToolCallUpdate]],
    response: Union[Response, AsyncResponse],
):
    for tool_call in tool_calls:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {"error": "Invalid JSON in arguments"}

        response.add_tool_call(
            ToolCall(
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=arguments,
            )
        )


class _Shared:
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"
    secondary_key_env_var = "GITHUB_TOKEN"
    can_stream = True

    def __init__(
        self,
        model_id: str,
        github_id: str,
        model_name: str,
        supports_schema: bool = False,
        requires_usage_stream_option: bool = True,
        supports_tools: bool = False,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
    ):
        self.model_id = f"github/{model_id}"
        self.github_id = github_id
        self.model_name = model_name
        self.supports_schema = supports_schema
        self.supports_tools = supports_tools
        self.attachment_types = set()
        if input_modalities and "image" in input_modalities:
            self.attachment_types.update(IMAGE_ATTACHMENTS)
        if input_modalities and "audio" in input_modalities:
            self.attachment_types.update(AUDIO_ATTACHMENTS)

        self.input_modalities = input_modalities
        self.output_modalities = output_modalities

        self.client_kwargs = {}
        # Use latest version
        self.client_kwargs["api_version"] = "2025-03-01-preview"

        self.streaming_model_extras = {}
        if requires_usage_stream_option:
            self.streaming_model_extras["stream_options"] = {
                "include_usage": True,
            }

    # Using the same display string for both the sync and async models
    # makes them not show up twice in `llm models`
    def __str__(self) -> str:
        return f"GitHub Models: {self.model_id}"

    def get_tools(self, prompt: Prompt) -> Optional[List[ChatCompletionsToolDefinition]]:
        if not self.supports_tools or not prompt.tools:
            return None

        return [
            ChatCompletionsToolDefinition(
                function=FunctionDefinition(
                    name=t.name,
                    description=t.description or None,
                    parameters=t.input_schema,
                ),
            )
            for t in prompt.tools
        ]

    def get_github_key(self, configured_key: Optional[str] = None) -> str:
        if configured_key is not None:
            # Someone already set model.key='...'
            return configured_key

        # Attempt to load a key using llm.get_key()
        key_value = llm.get_key(
            key_alias=self.needs_key,
            env_var=self.key_env_var,
        )
        if key_value:
            return key_value
        # Try secondary key
        if os.environ.get(self.secondary_key_env_var):
            return os.environ[self.secondary_key_env_var]

        # Show a useful error message
        message = f"""No key found - add one using 'llm keys set {self.needs_key}' \
or set the {self.key_env_var} or {self.secondary_key_env_var} environment variables"""
        raise llm.NeedsKeyException(message)


class GitHubModels(_Shared, Model):
    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
        # unset keys are handled by llm.Model.get_key()
        key: str = self.get_github_key(self.key)  # type: ignore

        with ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
            model=self.github_id,
            **self.client_kwargs,
        ) as client:
            response_format = "text"
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

            usage: Optional[CompletionsUsage] = None
            messages = build_messages(prompt, conversation)

            tools = self.get_tools(prompt)

            if stream:
                completion = client.complete(
                    messages=messages,
                    stream=True,
                    response_format=response_format,
                    model_extras=self.streaming_model_extras,
                    tools=tools,
                )
                tool_calls = {}

                for chunk in completion:
                    usage = usage or chunk.usage

                    if len(chunk.choices) == 0:
                        continue

                    delta = chunk.choices[0].delta
                    content = delta.content
                    append_streaming_tool_calls(tool_calls, delta)

                    if content is not None:
                        yield content

                add_tool_calls(
                    tool_calls.values(),
                    response,
                )

                response.response_json = None  # TODO
            else:
                completion = client.complete(
                    messages=messages,
                    stream=False,
                    response_format=response_format,
                    tools=tools,
                )
                usage = completion.usage

                tool_calls = completion.choices[0].message.tool_calls or []
                add_tool_calls(tool_calls, response)

                response.response_json = None  # TODO
                if completion.choices[0].message.content:
                    yield completion.choices[0].message.content

            if usage is not None:
                set_usage(usage, response)


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
            model=self.github_id,
            **self.client_kwargs,
        ) as client:
            response_format = "text"
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

            usage: Optional[CompletionsUsage] = None
            messages = build_messages(prompt, conversation)

            tools = self.get_tools(prompt)

            if stream:
                completion = await client.complete(
                    messages=messages,
                    stream=True,
                    response_format=response_format,
                    model_extras=self.streaming_model_extras,
                    tools=tools,
                )

                tool_calls = {}
                async for chunk in completion:
                    usage = usage or chunk.usage

                    if len(chunk.choices) == 0:
                        continue

                    delta = chunk.choices[0].delta
                    content = delta.content
                    append_streaming_tool_calls(tool_calls, delta)

                    if content is not None:
                        yield content

                add_tool_calls(
                    tool_calls.values(),
                    response,
                )

                response.response_json = None  # TODO
            else:
                completion = await client.complete(
                    messages=messages,
                    stream=False,
                    response_format=response_format,
                    tools=tools,
                )
                usage = usage or completion.usage

                tool_calls = completion.choices[0].message.tool_calls or []
                add_tool_calls(tool_calls, response)

                response.response_json = None  # TODO
                if completion.choices[0].message.content:
                    yield completion.choices[0].message.content

            if usage is not None:
                set_usage(usage, response)


class GitHubEmbeddingModel(EmbeddingModel):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"
    batch_size = 100

    def __init__(
        self, model_id: str, github_id: str, model_name: str, dimensions: Optional[int] = None
    ):
        self.model_id = f"github/{model_id}"
        self.github_id = github_id
        self.model_name = model_name
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
            "model": self.github_id,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embed(**kwargs)
        return ([float(x) for x in item.embedding] for item in response.data)
