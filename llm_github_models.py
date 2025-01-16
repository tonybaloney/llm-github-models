import llm
from llm.models import Conversation, Prompt, Response
from llm.default_plugins.openai_models import _attachment
from typing import Optional, Iterator, List, Dict, Any

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

CHAT_MODELS = [
 ('AI21-Jamba-1-5-Large', False, ['text'], ['text']),
 ('AI21-Jamba-1-5-Mini', False, ['text'], ['text']),
 ('Codestral-2501', False, ['text'], ['text']),
 ('Cohere-command-r', False, ['text'], ['text']),
 ('Cohere-command-r-08-2024', False, ['text'], ['text']),
 ('Cohere-command-r-plus', False, ['text'], ['text']),
 ('Cohere-command-r-plus-08-2024', False, ['text'], ['text']),
 ('Llama-3-2-11B-Vision-Instruct', False, ['text', 'image', 'audio'], ['text']),
 ('Llama-3-2-90B-Vision-Instruct', False, ['text', 'image', 'audio'], ['text']),
 ('Llama-3-3-70B-Instruct', False, ['text'], ['text']),
 ('Meta-Llama-3-1-405B-Instruct', False, ['text'], ['text']),
 ('Meta-Llama-3-1-70B-Instruct', False, ['text'], ['text']),
 ('Meta-Llama-3-1-8B-Instruct', False, ['text'], ['text']),
 ('Meta-Llama-3-70B-Instruct', False, ['text'], ['text']),
 ('Meta-Llama-3-8B-Instruct', False, ['text'], ['text']),
 ('Ministral-3B', False, ['text'], ['text']),
 ('Mistral-Large-2411', False, ['text'], ['text']),
 ('Mistral-Nemo', False, ['text'], ['text']),
 ('Mistral-large', False, ['text'], ['text']),
 ('Mistral-large-2407', False, ['text'], ['text']),
 ('Mistral-small', False, ['text'], ['text']),
 ('Phi-3-5-MoE-instruct', False, ['text'], ['text']),
 ('Phi-3-5-mini-instruct', False, ['text'], ['text']),
 ('Phi-3-5-vision-instruct', False, ['text', 'image'], []),
 ('Phi-3-medium-128k-instruct', False, ['text'], ['text']),
 ('Phi-3-medium-4k-instruct', False, ['text'], ['text']),
 ('Phi-3-mini-128k-instruct', False, ['text'], ['text']),
 ('Phi-3-mini-4k-instruct', False, ['text'], ['text']),
 ('Phi-3-small-128k-instruct', False, ['text'], ['text']),
 ('Phi-3-small-8k-instruct', False, ['text'], ['text']),
 ('Phi-4', False, ['text'], ['text']),
 ('gpt-4o', False, ['text', 'image', 'audio'], ['text']),
 ('gpt-4o-mini', False, ['text', 'image', 'audio'], ['text']),
 ('jais-30b-chat', False, ['text'], ['text']),
 ('o1', True, ['text', 'image'], ['text']),
 ('o1-mini', True, ['text'], ['text']),
 ('o1-preview', True, ['text'], ['text'])]

EMBEDDING_MODELS = ['Cohere-embed-v3-english',
 'Cohere-embed-v3-multilingual',
 'text-embedding-3-large',
 'text-embedding-3-small']

@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for model_id, can_stream, input_modalities, output_modalities in CHAT_MODELS:
        register(GitHubModels(model_id, can_stream=can_stream))


class _Shared:
    def build_messages(self, prompt: Prompt, conversation: Optional[Conversation]) -> List[Dict[str, Any]]:
        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment))
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt}
                    )
                messages.append(
                    {"role": "assistant", "content": prev_response.text_or_raise()}
                )
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        if not prompt.attachments:
            messages.append({"role": "user", "content": prompt.prompt})
        else:
            attachment_message = []
            if prompt.prompt:
                attachment_message.append({"type": "text", "text": prompt.prompt})
            for attachment in prompt.attachments:
                attachment_message.append(_attachment(attachment))
            messages.append({"role": "user", "content": attachment_message})
        return messages


class GitHubModels(_Shared, llm.Model):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"

    def __init__(self, model_id: str, can_stream: bool, input_modalities=None, output_modalities=None):
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.can_stream = can_stream
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

        client = ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
            model=self.model_name,
        )
        messages = self.build_messages(prompt, conversation)
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
            response.response_json = None # TODO
        else:
            completion = client.complete(
                messages=messages,
                stream=False,
            )
            response.response_json = None # TODO
            yield completion.choices[0].message.content
