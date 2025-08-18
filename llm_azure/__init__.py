try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    HAS_AZURE_PROJECTS_SDK = True
except ImportError:
    HAS_AZURE_PROJECTS_SDK = False

import logging

import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

logging.basicConfig(level=logging.WARNING)


@llm.hookimpl
def register_models(register):
    if not HAS_AZURE_PROJECTS_SDK:
        logging.debug("Azure Projects SDK is not available.")
        return

    endpoint = llm.get_key("azure.endpoint")

    with DefaultAzureCredential(exclude_interactive_browser_credential=False) as credential:  # pyright: ignore[reportPossiblyUnboundVariable]
        with AIProjectClient(endpoint=endpoint, credential=credential) as project_client:  # pyright: ignore[reportPossiblyUnboundVariable]
            for deployment in project_client.deployments.list():
                logging.info(deployment)
                register(
                    AzureAIFoundryModel(
                        deployment_name=deployment["name"],
                        client=project_client.get_openai_client(api_version="2025-04-01-preview"),
                    ),
                    AsyncAzureAIFoundryModel(
                        deployment_name=deployment["name"],
                        client=project_client.get_openai_client(api_version="2025-04-01-preview"),
                    ),
                )


class AzureAIFoundryModel(Chat):
    needs_key = None

    def __init__(self, deployment_name: str, client):
        self._client = client
        self.model_name = deployment_name
        self.model_id = "azure/" + deployment_name

    def __str__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"Azure AI Foundry: {self.model_id}"

    def get_client(self, key, *, async_=False):
        return self._client


class AsyncAzureAIFoundryModel(AsyncChat):
    needs_key = None

    def __init__(self, deployment_name: str, client):
        self._client = client
        self.model_name = deployment_name
        self.model_id = "azure/" + deployment_name

    def __str__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"Azure AI Foundry: {self.model_id}"

    def get_client(self, key, *, async_=False):
        return self._client
