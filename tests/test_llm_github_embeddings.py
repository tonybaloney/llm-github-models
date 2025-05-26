from unittest.mock import patch

import pytest
from azure.ai.inference.models import EmbeddingItem, EmbeddingsResult

from llm_github_models import EMBEDDING_MODELS, GitHubEmbeddingModel

EMBEDDING_MODEL_IDS = [f"github/{model}" for model in EMBEDDING_MODELS]


@pytest.mark.parametrize("model_id", EMBEDDING_MODELS)
def test_embedding_model_initialization(model_id: str):
    """Test that embedding models are initialized correctly."""
    embedding_model = GitHubEmbeddingModel(model_id)
    assert embedding_model.model_id == f"github/{model_id}"
    assert embedding_model.model_name == model_id


@patch("llm_github_models.EmbeddingsClient", autospec=True)
def test_embed_single_text(MockEmbeddingsClient):
    """Test embedding a single text."""
    # Setup mock
    mock_instance = MockEmbeddingsClient.return_value

    # Mock the response
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_embedding_item = EmbeddingItem(embedding=mock_embedding, index=0)
    mock_result = EmbeddingsResult(data=[mock_embedding_item])
    mock_instance.embed.return_value = mock_result

    # Create model and call embed
    model = GitHubEmbeddingModel("test-model")
    # Patch the get_key method to avoid actual key retrieval
    with patch.object(model, "get_key", return_value="test-key"):
        result = model.embed_batch(["This is a test text"])

    # Assertions
    MockEmbeddingsClient.assert_called_once()
    mock_instance.embed.assert_called_once_with(
        model="test-model",
        input=["This is a test text"],
    )

    result = list(result)
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3, 0.4, 0.5]


@patch("llm_github_models.EmbeddingsClient", autospec=True)
def test_embed_with_dimensions(MockEmbeddingsClient):
    """Test embedding with a custom dimensions."""
    # Setup mock
    mock_instance = MockEmbeddingsClient.return_value

    # Mock the response
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_embedding_item = EmbeddingItem(embedding=mock_embedding, index=0)
    mock_result = EmbeddingsResult(data=[mock_embedding_item])
    mock_instance.embed.return_value = mock_result

    # Create model and call embed
    model = GitHubEmbeddingModel("test-model", 1234)
    # Patch the get_key method to avoid actual key retrieval
    with patch.object(model, "get_key", return_value="test-key"):
        result = model.embed_batch(["This is a test text"])

    # Assertions
    MockEmbeddingsClient.assert_called_once()
    mock_instance.embed.assert_called_once_with(
        model="test-model",
        input=["This is a test text"],
        dimensions=1234,
    )

    result = list(result)
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3, 0.4, 0.5]


@patch("llm_github_models.EmbeddingsClient", autospec=True)
def test_embed_multiple_texts(MockEmbeddingsClient):
    """Test embedding multiple texts."""
    # Setup mock
    mock_instance = MockEmbeddingsClient.return_value

    # Mock the response for multiple embeddings
    mock_embedding1 = [0.1, 0.2, 0.3]
    mock_embedding2 = [0.4, 0.5, 0.6]

    mock_embedding_item1 = EmbeddingItem(embedding=mock_embedding1, index=0)
    mock_embedding_item2 = EmbeddingItem(embedding=mock_embedding2, index=1)

    mock_result = EmbeddingsResult(data=[mock_embedding_item1, mock_embedding_item2])

    mock_instance.embed.return_value = mock_result

    # Create model and call embed
    model = GitHubEmbeddingModel("test-model")
    # Patch the get_key method to avoid actual key retrieval
    with patch.object(model, "get_key", return_value="test-key"):
        texts = ["First text", "Second text"]
        result = model.embed_batch(texts)

    # Assertions
    MockEmbeddingsClient.assert_called_once()
    mock_instance.embed.assert_called_once_with(
        model="test-model",
        input=texts,
    )
    result = list(result)
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


@patch("llm_github_models.EmbeddingsClient", autospec=True)
def test_embed_empty_list(MockEmbeddingsClient):
    model = GitHubEmbeddingModel("text-embedding-3-small")
    with patch.object(model, "get_key", return_value="key"):
        result = model.embed_batch([])
    assert list(result) == []

    MockEmbeddingsClient.assert_not_called()


def test_register_embedding_models():
    registered = []

    def fake_register(instance):
        registered.append(instance)

    from llm_github_models import register_embedding_models

    register_embedding_models(fake_register)

    def check_model(model_id, dimensions=None):
        suffix = f"-{dimensions}" if dimensions else ""
        m = next(m for m in registered if m.model_id == f"github/{model_id}{suffix}")

        assert isinstance(m, GitHubEmbeddingModel)
        assert m.model_name == model_id
        assert m.dimensions == dimensions

        registered.remove(m)

    for model_id, supported_dimensions in EMBEDDING_MODELS:
        check_model(model_id)

        for dims in supported_dimensions:
            check_model(model_id, dims)

    assert not registered, "More models registered than expected"
