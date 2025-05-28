import json
import pathlib
from unittest.mock import Mock, patch

import pytest
from azure.ai.inference.models import (
    AudioContentItem,
    CompletionsUsage,
    ImageContentItem,
    ImageUrl,
    InputAudio,
    StreamingChatChoiceUpdate,
    StreamingChatCompletionsUpdate,
    SystemMessage,
    UserMessage,
)
from llm import get_async_model, get_model
from llm.models import Attachment, Conversation, Prompt, Response
from pydantic import BaseModel

from llm_github_models import GitHubModels, build_messages, set_usage

MODELS = ["github/gpt-4.1-mini", "github/gpt-4o-mini", "github/Llama-3.2-11B-Vision-Instruct"]


@pytest.mark.parametrize("model", MODELS)
def test_build_messages_no_conversation(model: str):
    # Test build_messages with conversation=None and a basic prompt without system.
    dummy_prompt = Prompt(prompt="Hello from prompt", system=None, attachments=[], model=model)
    messages = build_messages(dummy_prompt, None)
    # Should add one UserMessage from prompt since conversation is None.
    assert isinstance(messages, list)
    # Expecting only one message: UserMessage with content "Hello from prompt"
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, UserMessage)
    # For a simple user message, content is stored in 'content'
    # Compare against expected message content.
    assert msg.content == "Hello from prompt"


@pytest.mark.parametrize("model", MODELS)
def test_build_messages_with_conversation_no_prompt_system(model: str):
    # Create a dummy conversation with one response.
    dummy_prompt = Prompt(prompt="Hello from prompt", system=None, attachments=[], model=model)
    _model = get_model(model)
    # The response has a system message and a user message.
    dummy_response = Response(
        prompt=Prompt(prompt="Hello from last time", system=None, attachments=[], model=model),
        model=_model,
        stream=False,
    )
    dummy_convo = Conversation(responses=[dummy_response], model=_model)
    # Create a prompt with no system and without attachments.
    messages = build_messages(dummy_prompt, dummy_convo)
    assert len(messages) == 3


@pytest.mark.parametrize("model", MODELS)
def test_build_messages_with_conversation_prompt_system(model: str):
    # Create a dummy conversation with one response.
    dummy_prompt = Prompt(
        prompt="Hello from prompt", system="You are a hawk", attachments=[], model=model
    )
    _model = get_model(model)
    # The response has a system message and a user message.
    dummy_response = Response(
        prompt=Prompt(
            prompt="Hello from last time",
            system="You are a hawk",
            attachments=[],
            model=model,
        ),
        model=_model,
        stream=False,
    )
    dummy_convo = Conversation(responses=[dummy_response], model=_model)
    # Create a prompt with no system and without attachments.
    messages = build_messages(dummy_prompt, dummy_convo)
    assert len(messages) == 4
    # First message should be a system message.
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "You are a hawk"


def test_build_messages_with_image_path_attachment():
    # Create a dummy attachment object for an image.
    model: str = "gpt-4o"
    attachment = Attachment(
        path=pathlib.Path("tests/files/salmon.jpeg"), url=None, type="image/jpeg"
    )
    dummy_attachment = attachment
    # Create a prompt with an attachment and prompt text.
    dummy_prompt = Prompt(
        prompt="Here is an image:",
        system=None,
        model=model,
        attachments=[dummy_attachment],
    )
    # No conversation provided.
    messages = build_messages(dummy_prompt, None)
    # For a prompt with attachments, build_messages creates one UserMessage whose content is a list.
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, UserMessage)
    # The content should be a list with two items: TextContentItem and ImageContentItem.
    # Validate type and content.
    content_list = msg.content
    assert isinstance(content_list, list)
    assert len(content_list) == 2
    image_item = content_list[1]
    assert isinstance(image_item, ImageContentItem)
    # Check that image_item.image_url is an ImageUrl with the correct url.
    assert isinstance(image_item.image_url, ImageUrl)
    assert image_item.image_url.url.startswith("data:image/jpeg;base64,")


def test_build_messages_with_image_url_attachments():
    # Create a dummy attachment object for an image.
    model: str = "gpt-4o"
    attachment = Attachment(path=None, url="http://dummy.image/url.png", type="image/png")
    dummy_attachment = attachment
    # Create a prompt with an attachment and prompt text.
    dummy_prompt = Prompt(
        prompt="Here is an image:",
        system=None,
        model=model,
        attachments=[dummy_attachment],
    )
    # No conversation provided.
    messages = build_messages(dummy_prompt, None)
    # For a prompt with attachments, build_messages creates one UserMessage whose content is a list.
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, UserMessage)
    # The content should be a list with two items: TextContentItem and ImageContentItem.
    # Validate type and content.
    content_list = msg.content
    assert isinstance(content_list, list)
    assert len(content_list) == 2
    image_item = content_list[1]
    assert isinstance(image_item, ImageContentItem)
    # Check that image_item.image_url is an ImageUrl with the correct url.
    assert isinstance(image_item.image_url, ImageUrl)
    assert image_item.image_url.url == "http://dummy.image/url.png"


def test_build_messages_with_audio_path_attachment():
    # Create a dummy attachment object for an image.
    model: str = "gpt-4o"
    attachment = Attachment(path=pathlib.Path("tests/files/kick.wav"), url=None, type="audio/wav")
    dummy_attachment = attachment
    # Create a prompt with an attachment and prompt text.
    dummy_prompt = Prompt(
        prompt="Here is an audio clip:",
        system=None,
        model=model,
        attachments=[dummy_attachment],
    )
    # No conversation provided.
    messages = build_messages(dummy_prompt, None)
    # For a prompt with attachments, build_messages creates one UserMessage whose content is a list.
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, UserMessage)
    # The content should be a list with two items: TextContentItem and ImageContentItem.
    # Validate type and content.
    content_list = msg.content
    assert isinstance(content_list, list)
    assert len(content_list) == 2
    audio_item = content_list[1]
    assert isinstance(audio_item, AudioContentItem)
    # Check that image_item.image_url is an ImageUrl with the correct url.
    assert isinstance(audio_item.input_audio, InputAudio)
    assert audio_item.input_audio.data.startswith("UklGRuwiAAB")
    assert audio_item.input_audio.format == "wav"
    assert audio_item.input_audio.data.endswith("AAAAA=")


class DogSchema(BaseModel):
    """
    A schema for a dog with a name and age.
    """

    name: str
    age: int
    one_sentence_bio: str


def test_schema_with_unsupported_model():
    """
    Test that requesting a schema for an unsupported model raises an error.
    """
    model = get_model("github/Mistral-Nemo")

    with pytest.raises(ValueError):
        model.prompt("Invent a good dog", schema=DogSchema)


def test_schema_with_supported_model():
    """
    Test that requesting a schema for a supported model works.
    """
    model = get_model("github/gpt-4.1-mini")

    response = model.prompt("Invent a good dog named Buddy", schema=DogSchema)
    dog = json.loads(response.text())
    assert dog["name"] == "Buddy"


@pytest.mark.asyncio
async def test_async_model_prompt():
    """
    Test that the async model prompt works correctly.
    """
    model = get_async_model("github/gpt-4.1-mini")
    response = await model.prompt("What is the capital of France?")
    assert "Paris" in await response.text()


@patch("llm_github_models.ChatCompletionsClient", autospec=True)
def test_doesnt_request_streaming_usage_when_not_required(MockChatCompletionsClient):
    # Setup mock
    mock_update = StreamingChatCompletionsUpdate(
        {
            "choices": [StreamingChatChoiceUpdate({"delta": {"content": "Paris"}})],
        }
    )

    # `with ChatCompletionsClient(...) as client:`
    mock_instance = MockChatCompletionsClient.return_value.__enter__.return_value

    # `for chunk in client.complete(...)`
    mock_instance.complete.return_value.__iter__.return_value = [mock_update]

    model = GitHubModels("test-model", requires_usage_stream_option=False)

    # Patch the get_key method to avoid actual key retrieval
    with patch.object(model, "get_key", return_value="test-key"):
        result = model.prompt("What is the capital of France", stream=True)

    assert result.text() == "Paris"

    # Assertions
    call_kwargs = mock_instance.complete.call_args.kwargs
    assert call_kwargs["model_extras"] == {}, (
        "model_extras should be empty when requires_usage_stream_option is False"
    )


def test_set_usage():
    usage = CompletionsUsage(
        {
            "completion_tokens": 10,
            "prompt_tokens": 5,
            "extra": {
                "value": 123,
                "inner_empty": {},
                "inner_zero": 0,
            },
            "other": "data",
            "zero": 0,
            "empty": {},
        }
    )

    captured_usage = {}

    def usage_callback(input=None, output=None, details=None):
        captured_usage["input"] = input
        captured_usage["output"] = output
        captured_usage["details"] = details

    mock_response = Mock(spec=Response)
    mock_response.set_usage.side_effect = usage_callback

    set_usage(usage, mock_response)

    assert captured_usage["input"] == 5
    assert captured_usage["output"] == 10

    # Everything that is 0 or empty should be filtered out.
    assert captured_usage["details"] == {
        "extra": {
            "value": 123,
        },
        "other": "data",
    }


def test_sync_returns_usage():
    """
    Test that the sync model returns usage information for streaming and non-streaming.
    """
    model = get_model("github/gpt-4.1-mini")

    response = model.prompt("What is the capital of France?")
    usage = response.usage()
    assert_has_usage(usage)

    response = model.prompt("What is the capital of France?", stream=True)
    usage = response.usage()
    assert_has_usage(usage)


@pytest.mark.asyncio
async def test_async_returns_usage():
    """
    Test that the async model returns usage information for streaming and non-streaming.
    """
    model = get_async_model("github/gpt-4.1-mini")

    response = await model.prompt("What is the capital of France?")
    usage = await response.usage()
    assert_has_usage(usage)

    response = await model.prompt("What is the capital of France?", stream=True)
    usage = await response.usage()
    assert_has_usage(usage)


def assert_has_usage(usage):
    """
    Helper function to assert that usage has input and output tokens.
    """
    assert usage is not None
    assert usage.input is not None, "Usage input should not be None"
    assert usage.input > 0, "Usage input should be greater than 0"
    assert usage.output is not None, "Usage output should not be None"
    assert usage.output > 0, "Usage output should be greater than 0"
