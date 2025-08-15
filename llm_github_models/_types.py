from typing import List

from pydantic import BaseModel


class ChatModelSpec(BaseModel):
    llm_id: str  # The model ID used in llm commands, e.g. `llm -m github/ai21-jamba-1.5-large`
    github_id: str  # The model ID used by GitHub API
    name: str  # The name of the model
    supports_schemas: bool
    supports_streaming: bool
    supports_tools: bool
    supported_input_modalities: List[str]
    supported_output_modalities: List[str]


class EmbeddingModelSpec(BaseModel):
    llm_id: str  # The model ID used in llm commands, e.g. `llm -m github/ai21-jamba-1.5-large`
    github_id: str  # The model ID used by GitHub API
    name: str  # The name of the model
    dimensions: List[int]
