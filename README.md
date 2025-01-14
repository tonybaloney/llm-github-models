# GitHub Models Plugin for LLM

This is a plugin for [llm](https://llm.datasette.io) that uses [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) via the Azure AI Inference SDK.

To set the API key, use the `llm keys set` command or use the `GITHUB_MODELS_KEY` environment variable.

To get an API key, create a PAT inside GitHub.

All model names are affixed with `github/` to distinguish the OpenAI ones from the builtin models.

## Installation

```default
$ llm install llm-github-models
```

## Example

```default
$ llm prompt 'top facts about cheese' -m github/mistral-large                                                                                                                
Sure, here are some interesting facts about cheese:

1. There are over 2000 types of cheese: The variety of cheese is vast, with different flavors, textures, and aromas. This is due to factors like the type of milk used, the aging process, and the specific bacteria and mold cultures involved.

2. Cheese is an ancient food: The earliest evidence of cheese-making dates back to around 6000 BC, found in ancient Polish sites.
```
