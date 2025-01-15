# GitHub Models Plugin for LLM
[![PyPI](https://img.shields.io/pypi/v/llm-github-models.svg)](https://pypi.org/project/llm-github-models/)
[![Changelog](https://img.shields.io/github/v/release/tonybaloney/llm-github-models?include_prereleases&label=changelog)](https://github.com/tonybaloney/llm-github-models/releases)


This is a plugin for [llm](https://llm.datasette.io) that uses [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) via the Azure AI Inference SDK.

## Installation

```default
$ llm install llm-github-models
```

## Usage

To set the API key, use the `llm keys set github` command or use the `GITHUB_MODELS_KEY` environment variable.

To get an API key, create a personal access token (PAT) inside [GitHub Settings](https://github.com/settings/tokens).

Learn about [rate limits here](https://docs.github.com/github-models/prototyping-with-ai-models#rate-limits)

All model names are affixed with `github/` to distinguish the OpenAI ones from the builtin models.

## Example

```default
$ llm prompt 'top facts about cheese' -m github/mistral-large                                                                                                                
Sure, here are some interesting facts about cheese:

1. There are over 2000 types of cheese: The variety of cheese is vast, with different flavors, textures, and aromas. This is due to factors like the type of milk used, the aging process, and the specific bacteria and mold cultures involved.

2. Cheese is an ancient food: The earliest evidence of cheese-making dates back to around 6000 BC, found in ancient Polish sites.
```

### Image attachments

Multi-modal vision models can accept image attachments using the [LLM attachments](https://llm.datasette.io/en/stable/usage.html#attachments) options:

```bash
llm -m github/Llama-3.2-11B-Vision-Instruct "Describe this image" -a https://static.simonwillison.net/static/2024/pelicans.jpg
```

Produces
```bash
This image depicts a dense gathering of pelicans, with the largest birds situated in the center, showcasing their light brown plumage and long, pointed beaks. The pelicans are standing on a rocky shoreline, with a serene body of water behind them, characterized by its pale blue hue and gentle ripples. In the background, a dark, rocky cliff rises, adding depth to the scene.

The overall atmosphere of the image exudes tranquility, with the pelicans seemingly engaging in a social gathering or feeding activity. The photograph's clarity and focus on the pelicans' behavior evoke a sense of observation and appreciation for the natural world.
```