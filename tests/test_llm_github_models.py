import llm
import pytest
from click.testing import CliRunner
from inline_snapshot import snapshot
from llm.cli import cli


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("github/mistral-small")
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == snapshot("Gully or Skipper")
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "content": "Gully or Skipper",
            "role": "assistant",
            "finish_reason": "stop",
            "usage": {"completion_tokens": 5, "prompt_tokens": 17, "total_tokens": 22},
            "object": "chat.completion.chunk",
            "model": "openai/gpt-4o",
            "created": 1731200404,
        }
    )


@pytest.mark.vcr
def test_llm_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = (
        "OpenRouter: openrouter/openai/gpt-3.5-turbo",
        "OpenRouter: openrouter/anthropic/claude-2",
    )
    for fragment in fragments:
        assert fragment in result.output
