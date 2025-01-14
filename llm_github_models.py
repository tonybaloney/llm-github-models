import llm
from llm.models import Conversation, Prompt, Response
from llm.default_plugins.openai_models import combine_chunks, _attachment
from llm.utils import remove_dict_none_values
from typing import Optional, Iterator, List, Dict, Any

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for model_id in [
        'github/gpt-4o',
        'github/gpt-4o-mini',
        'github/o1',
        'github/o1-mini',
        'github/o1-preview',
        'github/phi-3-5-moe-instruct',
        'github/phi-3-5-mini-instruct',
        'github/phi-3-5-vision-instruct',
        'github/phi-3-medium-128k-instruct',
        'github/phi-3-medium-4k-instruct',
        'github/phi-3-mini-128k-instruct',
        'github/phi-3-mini-4k-instruct',
        'github/phi-3-small-128k-instruct',
        'github/phi-3-small-8k-instruct',
        'github/ai21-jamba-1-5-large',
        'github/ai21-jamba-1-5-mini',
        'github/codestral-2501',
        'github/cohere-command-r',
        'github/cohere-command-r-08-2024',
        'github/cohere-command-r-plus',
        'github/cohere-command-r-plus-08-2024',
        'github/cohere-embed-v3-english',
        'github/cohere-embed-v3-multilingual',
        'github/llama-3-2-11b-vision-instruct',
        'github/llama-3-2-90b-vision-instruct',
        'github/llama-3-3-70b-instruct',
        'github/meta-llama-3-1-405b-instruct',
        'github/meta-llama-3-1-70b-instruct',
        'github/meta-llama-3-1-8b-instruct',
        'github/meta-llama-3-70b-instruct',
        'github/meta-llama-3-8b-instruct',
        'github/ministral-3b',
        'github/mistral-large-2411',
        'github/mistral-nemo',
        'github/mistral-large',
        'github/mistral-large-2407',
        'github/mistral-small',
        'github/jais-30b-chat',
    ]:
        register(GitHubModels(model_id))


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
    can_stream = True


    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]

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
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
        else:
            completion = client.complete(
                messages=messages,
                stream=False,
            )
            response.response_json = remove_dict_none_values(completion.model_dump())
            yield completion.choices[0].message.content
