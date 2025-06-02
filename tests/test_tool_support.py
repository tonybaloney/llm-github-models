import json

import pytest
from azure.ai.inference.models import StreamingChatResponseMessageUpdate
from llm import get_async_model, get_model

from llm_github_models import append_streaming_tool_calls


def test_model_supports_tools():
    model = get_model("github/gpt-4o-mini")
    assert model.supports_tools is True


def test_append_streaming_tool_calls():
    tool_calls = {}

    # call_1 => multiply
    append_streaming_tool_calls(
        tool_calls,
        StreamingChatResponseMessageUpdate(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "index": 0,
                        "function": {"name": "multiply", "arguments": ""},
                    }
                ]
            }
        ),
    )

    # call_1 => multiply(x:
    # call_2 => add(x: 1,
    append_streaming_tool_calls(
        tool_calls,
        StreamingChatResponseMessageUpdate(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": '{ "x": '},
                    },
                    {
                        "id": "call_2",
                        "index": 1,
                        "function": {"name": "add", "arguments": '{ "x": 1,'},
                    },
                ]
            }
        ),
    )

    # call_1 => multiply(x: 2, y: 3)
    # call_2 => add(x: 1, y:
    append_streaming_tool_calls(
        tool_calls,
        StreamingChatResponseMessageUpdate(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": '2, "y": 3}'},
                    },
                    {
                        "index": 1,
                        "function": {"name": "add", "arguments": ' "y":'},
                    },
                ]
            }
        ),
    )

    # call_1 => multiply(x: 2, y: 3)
    # call_2 => add(x: 1, y: 3)
    append_streaming_tool_calls(
        tool_calls,
        StreamingChatResponseMessageUpdate(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "function": {"name": "add", "arguments": " 3 }"},
                    },
                ]
            }
        ),
    )

    assert len(tool_calls) == 2

    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function.name == "multiply"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 2, "y": 3}

    assert tool_calls[1].id == "call_2"
    assert tool_calls[1].function.name == "add"
    assert json.loads(tool_calls[1].function.arguments) == {"x": 1, "y": 3}


@pytest.mark.parametrize("stream", [True, False])
def test_sync_uses_tools(stream):
    model = get_model("github/gpt-4o-mini")

    # Create a prompt with a tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    chain = model.chain("What is 34234 * 213345?", tools=[multiply], stream=stream).responses()  # type: ignore

    tool_call_resp = next(chain)

    tool_calls = tool_call_resp.tool_calls()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "multiply"
    assert tool_calls[0].arguments == {"x": 34234, "y": 213345}

    # Sometimes it likes to add commas to the output number
    response_text = next(chain).text().replace(",", "")
    assert "7303652730" in response_text


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_async_uses_tools(stream):
    model = get_async_model("github/gpt-4o-mini")

    # Create a prompt with a tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    chain = model.chain("What is 34234 * 213345?", tools=[multiply], stream=stream).responses()  # type: ignore

    responses = []
    async for resp in chain:
        responses.append(resp)

    tool_call_resp = responses[0]

    tool_calls = await tool_call_resp.tool_calls()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "multiply"
    assert tool_calls[0].arguments == {"x": 34234, "y": 213345}

    # Sometimes it likes to add commas to the output number
    response_text = (await responses[1].text()).replace(",", "")
    assert "7303652730" in response_text


def test_multi_tool_use():
    model = get_model("github/gpt-4o-mini")

    # Create a prompt with a tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    chain = model.chain(
        "What is (34234 * 213345) + (45345 * 324456)?",
        tools=[multiply, add],  # type: ignore
    ).responses()

    tool_calls = next(chain).tool_calls()
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "multiply"
    assert tool_calls[0].arguments == {"x": 34234, "y": 213345}

    assert tool_calls[1].name == "multiply"
    assert tool_calls[1].arguments == {"x": 45345, "y": 324456}

    tool_calls = next(chain).tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "add"
    assert tool_calls[0].arguments == {"x": 7303652730, "y": 14712457320}

    # Sometimes it likes to add commas to the output number
    response_text = next(chain).text().replace(",", "")
    assert "22016110050" in response_text
