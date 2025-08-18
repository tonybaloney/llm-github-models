from llm_github_models._types import EmbeddingModelSpec

EMBEDDING_MODELS = [
    EmbeddingModelSpec(
        llm_id="cohere-embed-v3-english",
        github_id="cohere/cohere-embed-v3-english",
        name="Cohere Embed v3 English",
        dimensions=None,
    ),
    EmbeddingModelSpec(
        llm_id="cohere-embed-v3-multilingual",
        github_id="cohere/cohere-embed-v3-multilingual",
        name="Cohere Embed v3 Multilingual",
        dimensions=None,
    ),
    EmbeddingModelSpec(
        llm_id="text-embedding-3-large-1024",
        github_id="openai/text-embedding-3-large",
        name="OpenAI Text Embedding 3 (large) (1024)",
        dimensions=1024,
    ),
    EmbeddingModelSpec(
        llm_id="text-embedding-3-large-256",
        github_id="openai/text-embedding-3-large",
        name="OpenAI Text Embedding 3 (large) (256)",
        dimensions=256,
    ),
    EmbeddingModelSpec(
        llm_id="text-embedding-3-small-512",
        github_id="openai/text-embedding-3-small",
        name="OpenAI Text Embedding 3 (small) (512)",
        dimensions=512,
    ),
]
