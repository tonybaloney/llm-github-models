import llm
from llm.models import Conversation, Prompt, Response
from llm.default_plugins.openai_models import _attachment
from typing import Optional, Iterator, List, Dict, Any

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for model_id, can_stream in [
        ('github/gpt-4o', True),        
        ('github/gpt-4o-mini', True),
        ('github/o1', False),
        ('github/o1-mini', False),
        ('github/o1-preview', False),
        ('github/phi-3-5-moe-instruct', True),
        ('github/phi-3-5-mini-instruct', True),
        ('github/phi-3-5-vision-instruct', True),
        ('github/phi-3-medium-128k-instruct', True),
        ('github/phi-3-medium-4k-instruct', True),
        ('github/phi-3-mini-128k-instruct', True),
        ('github/phi-3-mini-4k-instruct', True),
        ('github/phi-3-small-128k-instruct', True),
        ('github/phi-3-small-8k-instruct', True),
        ('github/ai21-jamba-1-5-large', True),
        ('github/ai21-jamba-1-5-mini', True),
        ('github/codestral-2501', True),
        ('github/cohere-command-r', True),
        ('github/cohere-command-r-08-2024', True),
        ('github/cohere-command-r-plus', True),
        ('github/cohere-command-r-plus-08-2024', True),
        ('github/cohere-embed-v3-english', True),
        ('github/cohere-embed-v3-multilingual', True),
        ('github/llama-3-2-11b-vision-instruct', True),
        ('github/llama-3-2-90b-vision-instruct', True),
        ('github/llama-3-3-70b-instruct', True),
        ('github/meta-llama-3-1-405b-instruct', True),
        ('github/meta-llama-3-1-70b-instruct', True),
        ('github/meta-llama-3-1-8b-instruct', True),
        ('github/meta-llama-3-70b-instruct', True),
        ('github/meta-llama-3-8b-instruct', True),
        ('github/ministral-3b', True),
        ('github/mistral-large-2411', True),
        ('github/mistral-nemo', True),
        ('github/mistral-large', True),
        ('github/mistral-large-2407', True),
        ('github/mistral-small', True),
        ('github/jais-30b-chat', True),
    ]:
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
    needs_key = True
    key_env_var = "GITHUB_MODELS_KEY"

    def __init__(self, model_id: str, can_stream: bool):
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.can_stream = can_stream

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
