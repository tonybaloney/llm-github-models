from llm_github_models import build_messages
from llm.models import Prompt, Conversation, Attachment, Response
from llm import get_model

from azure.ai.inference.models import (
    UserMessage,
    ImageContentItem,
    ImageUrl,
    AudioContentItem,
    InputAudio,
    SystemMessage,
)
import pytest
import pathlib

MODELS = ["github/gpt-4o", "github/gpt-4o-mini"]


@pytest.mark.parametrize("model", MODELS)
def test_build_messages_no_conversation(model: str):
    # Test build_messages with conversation=None and a basic prompt without system.
    dummy_prompt = Prompt(
        prompt="Hello from prompt", system=None, attachments=[], model=model
    )
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
    dummy_prompt = Prompt(
        prompt="Hello from prompt", system=None, attachments=[], model=model
    )
    _model = get_model(model)
    # The response has a system message and a user message.
    dummy_response = Response(
        prompt=Prompt(
            prompt="Hello from last time", system=None, attachments=[], model=model
        ),
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
    attachment = Attachment(
        path=None, url="http://dummy.image/url.png", type="image/png"
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
    assert image_item.image_url.url == "http://dummy.image/url.png"


def test_build_messages_with_audio_path_attachment():
    # Create a dummy attachment object for an image.
    model: str = "gpt-4o"
    attachment = Attachment(
        path=pathlib.Path("tests/files/kick.wav"), url=None, type="audio/wav"
    )
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
