from llm_github_models import build_messages
from llm.models import Prompt, Conversation, Attachment, Response
from llm import get_model

from azure.ai.inference.models import (
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
)
import pytest

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
def test_build_messages_with_attachments(model: str):
    # Create a dummy attachment object for an image.
    dummy_attachment = Attachment(
        path=None, url="http://dummy.image/url.png", type="image/png"
    )
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
    # First item should be TextContentItem with text "Here is an image:"
    text_item = content_list[0]
    assert isinstance(text_item, TextContentItem)
    assert text_item.text == "Here is an image:"
    # Second item should be an ImageContentItem with an ImageUrl having the specified url.
    image_item = content_list[1]
    assert isinstance(image_item, ImageContentItem)
    # Check that image_item.image_url is an ImageUrl with the correct url.
    assert isinstance(image_item.image_url, ImageUrl)
    assert image_item.image_url.url == "http://dummy.image/url.png"
