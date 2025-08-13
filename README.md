# GitHub Models Plugin for LLM
[![PyPI](https://img.shields.io/pypi/v/llm-github-models.svg)](https://pypi.org/project/llm-github-models/)
[![Changelog](https://img.shields.io/github/v/release/tonybaloney/llm-github-models?include_prereleases&label=changelog)](https://github.com/tonybaloney/llm-github-models/releases)

This is a plugin for [llm](https://llm.datasette.io) that uses [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) via the Azure AI Inference SDK. GitHub Models is available to all GitHub users and offers **free** usage of many AI LLMs. 

## Features

- Support for all >30 models, including GPT-4o, 4.1, o3, DeepSeek-R1, Llama3.x and more
- Support for [schemas](https://llm.datasette.io/en/stable/schemas.html)
- Output token usage
- Support for [Embedding Models](https://llm.datasette.io/en/stable/embeddings/index.html)
- Async and streaming outputs (model dependent)
- Support for model attachments
- Support for [tools](https://llm.datasette.io/en/stable/tools.html)

## Installation

```default
$ llm install llm-github-models
```

or `pip install llm-github-models`

## Usage

To set the API key, use the `llm keys set github` command or use the `GITHUB_MODELS_KEY` environment variable.
If neither are present, `GITHUB_TOKEN` will be used. This environment variable is set in both GitHub Actions and the GitHub CLI.

To get an API key, create a personal access token (PAT) inside [GitHub Settings](https://github.com/settings/tokens).

Learn about [rate limits here](https://docs.github.com/github-models/prototyping-with-ai-models#rate-limits)

All model names are affixed with `github/` to distinguish the OpenAI ones from the builtin models.

## Example

```default
$ llm prompt 'top facts about cheese' -m github/gpt-4.1-mini
Sure! Here are some top facts about cheese:

1. **Ancient Origins**: Cheese is one of the oldest man-made foods, with evidence of cheese-making dating back over 7,000 years.

2. **Variety**: There are over 1,800 distinct types of cheese worldwide, varying by texture, flavor, milk source, and production methods.
```

## Usage in GitHub Actions

By default, GitHub Actions runners have limited permissions, to generate a `GITHUB_TOKEN` with models access, configure a workflow with these settings:

```yaml
name: Python package

on:
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      models: read
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install "llm-github-models"
    - name: Run llm commands
      run: |
        llm prompt -m github/gpt-5-mini "Test prompt"
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

## Supported Models

### Chat Models

| Model Name | Schemas | Tools | Input Modalities | Output Modalities |
|------------|---------|-------|------------------|-------------------|
| AI21-Jamba-1.5-Large | ❌ | ❌ | text | text |
| AI21-Jamba-1.5-Mini | ❌ | ❌ | text | text |
| Codestral-2501 | ❌ | ✅ | text | text |
| Cohere-command-r | ❌ | ✅ | text | text |
| Cohere-command-r-08-2024 | ❌ | ✅ | text | text |
| Cohere-command-r-plus | ❌ | ✅ | text | text |
| Cohere-command-r-plus-08-2024 | ❌ | ✅ | text | text |
| DeepSeek-R1 | ❌ | ❌ | text | text |
| DeepSeek-R1-0528 | ❌ | ❌ | text | text |
| DeepSeek-V3 | ❌ | ❌ | text | text |
| DeepSeek-V3-0324 | ❌ | ❌ | text | text |
| Llama-3.2-11B-Vision-Instruct | ❌ | ❌ | text, image, audio | text |
| Llama-3.2-90B-Vision-Instruct | ❌ | ❌ | text, image, audio | text |
| Llama-3.3-70B-Instruct | ❌ | ❌ | text | text |
| Llama-4-Maverick-17B-128E-Instruct-FP8 | ❌ | ❌ | text, image | text |
| Llama-4-Scout-17B-16E-Instruct | ❌ | ❌ | text, image | text |
| MAI-DS-R1 | ❌ | ❌ | text | text |
| Meta-Llama-3-70B-Instruct | ❌ | ❌ | text | text |
| Meta-Llama-3-8B-Instruct | ❌ | ❌ | text | text |
| Meta-Llama-3.1-405B-Instruct | ❌ | ❌ | text | text |
| Meta-Llama-3.1-70B-Instruct | ❌ | ❌ | text | text |
| Meta-Llama-3.1-8B-Instruct | ❌ | ❌ | text | text |
| Ministral-3B | ❌ | ✅ | text | text |
| Mistral-Large-2411 | ❌ | ✅ | text | text |
| Mistral-Nemo | ❌ | ✅ | text | text |
| Mistral-large-2407 | ❌ | ✅ | text | text |
| Mistral-small | ❌ | ✅ | text | text |
| Phi-3-medium-128k-instruct | ❌ | ❌ | text | text |
| Phi-3-medium-4k-instruct | ❌ | ❌ | text | text |
| Phi-3-mini-128k-instruct | ❌ | ❌ | text | text |
| Phi-3-mini-4k-instruct | ❌ | ❌ | text | text |
| Phi-3-small-128k-instruct | ❌ | ❌ | text | text |
| Phi-3-small-8k-instruct | ❌ | ❌ | text | text |
| Phi-3.5-MoE-instruct | ❌ | ❌ | text | text |
| Phi-3.5-mini-instruct | ❌ | ❌ | text | text |
| Phi-3.5-vision-instruct | ❌ | ❌ | text, image | text |
| Phi-4 | ❌ | ❌ | text | text |
| Phi-4-mini-instruct | ❌ | ❌ | text | text |
| Phi-4-mini-reasoning | ❌ | ❌ | text | text |
| Phi-4-multimodal-instruct | ❌ | ❌ | audio, image, text | text |
| Phi-4-reasoning | ❌ | ❌ | text | text |
| cohere-command-a | ❌ | ✅ | text | text |
| gpt-4.1 | ✅ | ✅ | text, image | text |
| gpt-4.1-mini | ✅ | ✅ | text, image | text |
| gpt-4.1-nano | ✅ | ✅ | text, image | text |
| gpt-4o | ✅ | ✅ | text, image, audio | text |
| gpt-4o-mini | ✅ | ✅ | text, image, audio | text |
| gpt-5 | ✅ | ✅ | text, image | text |
| gpt-5-chat | ✅ | ✅ | text, image | text |
| gpt-5-mini | ✅ | ✅ | text, image | text |
| gpt-5-nano | ✅ | ✅ | text, image | text |
| grok-3 | ❌ | ✅ | text | text |
| grok-3-mini | ❌ | ✅ | text | text |
| jais-30b-chat | ❌ | ❌ | text | text |
| mistral-medium-2505 | ❌ | ✅ | text, image | text |
| mistral-small-2503 | ❌ | ✅ | text, image | text |
| o1 | ✅ | ✅ | text, image | text |
| o1-mini | ❌ | ❌ | text | text |
| o1-preview | ❌ | ❌ | text | text |
| o3 | ❌ | ✅ | text, image | text |
| o3-mini | ✅ | ✅ | text | text |
| o4-mini | ❌ | ✅ | text, image | text |

### AI21 Jamba 1.5 Large

Usage: `llm -m github/AI21-Jamba-1.5-Large`

**Publisher:** AI21 Labs 

**Description:** A 398B parameters (94B active) multilingual model, offering a 256K long context window, function calling, structured output, and grounded generation. 

### AI21 Jamba 1.5 Mini

Usage: `llm -m github/AI21-Jamba-1.5-Mini`

**Publisher:** AI21 Labs 

**Description:** A 52B parameters (12B active) multilingual model, offering a 256K long context window, function calling, structured output, and grounded generation. 

### Codestral 25.01

Usage: `llm -m github/Codestral-2501`

**Publisher:** Mistral AI 

**Description:** Codestral 25.01 by Mistral AI is designed for code generation, supporting 80+ programming languages, and optimized for tasks like code completion and fill-in-the-middle 

### Cohere Command R

Usage: `llm -m github/Cohere-command-r`

**Publisher:** Cohere 

**Description:** Command R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise. 

### Cohere Command R 08-2024

Usage: `llm -m github/Cohere-command-r-08-2024`

**Publisher:** Cohere 

**Description:** Command R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise. 

### Cohere Command R+

Usage: `llm -m github/Cohere-command-r-plus`

**Publisher:** Cohere 

**Description:** Command R+ is a state-of-the-art RAG-optimized model designed to tackle enterprise-grade workloads. 

### Cohere Command R+ 08-2024

Usage: `llm -m github/Cohere-command-r-plus-08-2024`

**Publisher:** Cohere 

**Description:** Command R+ is a state-of-the-art RAG-optimized model designed to tackle enterprise-grade workloads. 

### Cohere Embed v3 English

Usage: `llm -m github/Cohere-embed-v3-english`

**Publisher:** Cohere 

**Description:** Cohere Embed English is the market's leading text representation model used for semantic search, retrieval-augmented generation (RAG), classification, and clustering. 

### Cohere Embed v3 Multilingual

Usage: `llm -m github/Cohere-embed-v3-multilingual`

**Publisher:** Cohere 

**Description:** Cohere Embed Multilingual is the market's leading text representation model used for semantic search, retrieval-augmented generation (RAG), classification, and clustering. 

### DeepSeek-R1

Usage: `llm -m github/DeepSeek-R1`

**Publisher:** DeepSeek 

**Description:** DeepSeek-R1 excels at reasoning tasks using a step-by-step training process, such as language, scientific reasoning, and coding tasks. 

### DeepSeek-R1-0528

Usage: `llm -m github/DeepSeek-R1-0528`

**Publisher:** DeepSeek 

**Description:** The DeepSeek R1 0528 model has improved reasoning capabilities, this version also offers a reduced hallucination rate, enhanced support for function calling, and better experience for vibe coding. 

### DeepSeek-V3

Usage: `llm -m github/DeepSeek-V3`

**Publisher:** DeepSeek 

**Description:** A strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

### DeepSeek-V3-0324

Usage: `llm -m github/DeepSeek-V3-0324`

**Publisher:** DeepSeek 

**Description:** DeepSeek-V3-0324 demonstrates notable improvements over its predecessor, DeepSeek-V3, in several key aspects, including enhanced reasoning, improved function calling, and superior code generation capabilities. 

### FLUX1.1 [pro]

Usage: `llm -m github/Flux-1.1-Pro`

**Publisher:** Black Forest Labs 

**Description:** Generate images with amazing image quality, prompt adherence, and diversity at blazing fast speeds. FLUX1.1 [pro] delivers six times faster image generation and achieved the highest Elo score on Artificial Analysis benchmarks when launched, surpassing all  

### FLUX.1 Kontext [pro]

Usage: `llm -m github/Flux.1-Kontext-pro`

**Publisher:** Black Forest Labs 

**Description:** Generate and edit images through both text and image prompts. FLUX.1 Kontext is a multimodal flow matching model that enables both text-to-image generation and in-context image editing. Modify images while maintaining character consistency and performing l 

### Llama-3.2-11B-Vision-Instruct

Usage: `llm -m github/Llama-3.2-11B-Vision-Instruct`

**Publisher:** Meta 

**Description:** Excels in image reasoning capabilities on high-res images for visual understanding apps. 

### Llama-3.2-90B-Vision-Instruct

Usage: `llm -m github/Llama-3.2-90B-Vision-Instruct`

**Publisher:** Meta 

**Description:** Advanced image reasoning capabilities for visual understanding agentic apps. 

### Llama-3.3-70B-Instruct

Usage: `llm -m github/Llama-3.3-70B-Instruct`

**Publisher:** Meta 

**Description:** Llama 3.3 70B Instruct offers enhanced reasoning, math, and instruction following with performance comparable to Llama 3.1 405B. 

### Llama 4 Maverick 17B 128E Instruct FP8

Usage: `llm -m github/Llama-4-Maverick-17B-128E-Instruct-FP8`

**Publisher:** Meta 

**Description:** Llama 4 Maverick 17B 128E Instruct FP8 is great at precise image understanding and creative writing, offering high quality at a lower price compared to Llama 3.3 70B 

### Llama 4 Scout 17B 16E Instruct

Usage: `llm -m github/Llama-4-Scout-17B-16E-Instruct`

**Publisher:** Meta 

**Description:** Llama 4 Scout 17B 16E Instruct is great at multi-document summarization, parsing extensive user activity for personalized tasks, and reasoning over vast codebases. 

### MAI-DS-R1

Usage: `llm -m github/MAI-DS-R1`

**Publisher:** Microsoft 

**Description:** MAI-DS-R1 is a DeepSeek-R1 reasoning model that has been post-trained by the Microsoft AI team to fill in information gaps in the previous version of the model and improve its harm protections while maintaining R1 reasoning capabilities. 

### Meta-Llama-3-70B-Instruct

Usage: `llm -m github/Meta-Llama-3-70B-Instruct`

**Publisher:** Meta 

**Description:** A powerful 70-billion parameter model excelling in reasoning, coding, and broad language applications. 

### Meta-Llama-3-8B-Instruct

Usage: `llm -m github/Meta-Llama-3-8B-Instruct`

**Publisher:** Meta 

**Description:** A versatile 8-billion parameter model optimized for dialogue and text generation tasks. 

### Meta-Llama-3.1-405B-Instruct

Usage: `llm -m github/Meta-Llama-3.1-405B-Instruct`

**Publisher:** Meta 

**Description:** The Llama 3.1 instruction tuned text only models are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks. 

### Meta-Llama-3.1-70B-Instruct

Usage: `llm -m github/Meta-Llama-3.1-70B-Instruct`

**Publisher:** Meta 

**Description:** The Llama 3.1 instruction tuned text only models are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks. 

### Meta-Llama-3.1-8B-Instruct

Usage: `llm -m github/Meta-Llama-3.1-8B-Instruct`

**Publisher:** Meta 

**Description:** The Llama 3.1 instruction tuned text only models are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks. 

### Ministral 3B

Usage: `llm -m github/Ministral-3B`

**Publisher:** Mistral AI 

**Description:** Ministral 3B is a state-of-the-art Small Language Model (SLM) optimized for edge computing and on-device applications. As it is designed for low-latency and compute-efficient inference, it it also the perfect model for standard GenAI applications that have 

### Mistral Large 24.11

Usage: `llm -m github/Mistral-Large-2411`

**Publisher:** Mistral AI 

**Description:** Mistral Large 24.11 offers enhanced system prompts, advanced reasoning and function calling capabilities. 

### Mistral Nemo

Usage: `llm -m github/Mistral-Nemo`

**Publisher:** Mistral AI 

**Description:** Mistral Nemo is a cutting-edge Language Model (LLM) boasting state-of-the-art reasoning, world knowledge, and coding capabilities within its size category. 

### Mistral Large (2407)

Usage: `llm -m github/Mistral-large-2407`

**Publisher:** Mistral AI 

**Description:** Mistral Large (2407) is an advanced Large Language Model (LLM) with state-of-the-art reasoning, knowledge and coding capabilities. 

### Mistral Small

Usage: `llm -m github/Mistral-small`

**Publisher:** Mistral AI 

**Description:** Mistral Small can be used on any language-based task that requires high efficiency and low latency. 

### Phi-3-medium instruct (128k)

Usage: `llm -m github/Phi-3-medium-128k-instruct`

**Publisher:** Microsoft 

**Description:** Same Phi-3-medium model, but with a larger context size for RAG or few shot prompting. 

### Phi-3-medium instruct (4k)

Usage: `llm -m github/Phi-3-medium-4k-instruct`

**Publisher:** Microsoft 

**Description:** A 14B parameters model, proves better quality than Phi-3-mini, with a focus on high-quality, reasoning-dense data. 

### Phi-3-mini instruct (128k)

Usage: `llm -m github/Phi-3-mini-128k-instruct`

**Publisher:** Microsoft 

**Description:** Same Phi-3-mini model, but with a larger context size for RAG or few shot prompting. 

### Phi-3-mini instruct (4k)

Usage: `llm -m github/Phi-3-mini-4k-instruct`

**Publisher:** Microsoft 

**Description:** Tiniest member of the Phi-3 family. Optimized for both quality and low latency. 

### Phi-3-small instruct (128k)

Usage: `llm -m github/Phi-3-small-128k-instruct`

**Publisher:** Microsoft 

**Description:** Same Phi-3-small model, but with a larger context size for RAG or few shot prompting. 

### Phi-3-small instruct (8k)

Usage: `llm -m github/Phi-3-small-8k-instruct`

**Publisher:** Microsoft 

**Description:** A 7B parameters model, proves better quality than Phi-3-mini, with a focus on high-quality, reasoning-dense data. 

### Phi-3.5-MoE instruct (128k)

Usage: `llm -m github/Phi-3.5-MoE-instruct`

**Publisher:** Microsoft 

**Description:** A new mixture of experts model 

### Phi-3.5-mini instruct (128k)

Usage: `llm -m github/Phi-3.5-mini-instruct`

**Publisher:** Microsoft 

**Description:** Refresh of Phi-3-mini model. 

### Phi-3.5-vision instruct (128k)

Usage: `llm -m github/Phi-3.5-vision-instruct`

**Publisher:** Microsoft 

**Description:** Refresh of Phi-3-vision model. 

### Phi-4

Usage: `llm -m github/Phi-4`

**Publisher:** Microsoft 

**Description:** Phi-4 14B, a highly capable model for low latency scenarios. 

### Phi-4-mini-instruct

Usage: `llm -m github/Phi-4-mini-instruct`

**Publisher:** Microsoft 

**Description:** 3.8B parameters Small Language Model outperforming larger models in reasoning, math, coding, and function-calling 

### Phi-4-mini-reasoning

Usage: `llm -m github/Phi-4-mini-reasoning`

**Publisher:** Microsoft 

**Description:** Lightweight math reasoning model optimized for multi-step problem solving 

### Phi-4-multimodal-instruct

Usage: `llm -m github/Phi-4-multimodal-instruct`

**Publisher:** Microsoft 

**Description:** First small multimodal model to have 3 modality inputs (text, audio, image), excelling in quality and efficiency 

### Phi-4-Reasoning

Usage: `llm -m github/Phi-4-reasoning`

**Publisher:** Microsoft 

**Description:** State-of-the-art open-weight reasoning model. 

### Cohere Command A

Usage: `llm -m github/cohere-command-a`

**Publisher:** Cohere 

**Description:** Command A is a highly efficient generative model that excels at agentic and multilingual use cases. 

### Cohere Embed 4

Usage: `llm -m github/embed-v-4-0`

**Publisher:** Cohere 

**Description:** Embed 4 transforms texts and images into numerical vectors 

### OpenAI GPT-4.1

Usage: `llm -m github/gpt-4.1`

**Publisher:** OpenAI 

**Description:** gpt-4.1 outperforms gpt-4o across the board, with major gains in coding, instruction following, and long-context understanding 

### OpenAI GPT-4.1-mini

Usage: `llm -m github/gpt-4.1-mini`

**Publisher:** OpenAI 

**Description:** gpt-4.1-mini outperform gpt-4o-mini across the board, with major gains in coding, instruction following, and long-context handling 

### OpenAI GPT-4.1-nano

Usage: `llm -m github/gpt-4.1-nano`

**Publisher:** OpenAI 

**Description:** gpt-4.1-nano provides gains in coding, instruction following, and long-context handling along with lower latency and cost 

### OpenAI GPT-4o

Usage: `llm -m github/gpt-4o`

**Publisher:** OpenAI 

**Description:** OpenAI's most advanced multimodal model in the gpt-4o family. Can handle both text and image inputs. 

### OpenAI GPT-4o mini

Usage: `llm -m github/gpt-4o-mini`

**Publisher:** OpenAI 

**Description:** An affordable, efficient AI solution for diverse text and image tasks. 

### OpenAI gpt-5

Usage: `llm -m github/gpt-5`

**Publisher:** OpenAI 

**Description:** gpt-5 is designed for logic-heavy and multi-step tasks.  

### OpenAI gpt-5-chat (preview)

Usage: `llm -m github/gpt-5-chat`

**Publisher:** OpenAI 

**Description:** gpt-5-chat (preview) is an advanced, natural, multimodal, and context-aware conversations for enterprise applications. 

### OpenAI gpt-5-mini

Usage: `llm -m github/gpt-5-mini`

**Publisher:** OpenAI 

**Description:** gpt-5-mini is a lightweight version for cost-sensitive applications. 

### OpenAI gpt-5-nano

Usage: `llm -m github/gpt-5-nano`

**Publisher:** OpenAI 

**Description:** gpt-5-nano is optimized for speed, ideal for applications requiring low latency.  

### Grok 3

Usage: `llm -m github/grok-3`

**Publisher:** xAI 

**Description:** Grok 3 is xAI's debut model, pretrained by Colossus at supermassive scale to excel in specialized domains like finance, healthcare, and the law. 

### Grok 3 Mini

Usage: `llm -m github/grok-3-mini`

**Publisher:** xAI 

**Description:** Grok 3 Mini is a lightweight model that thinks before responding. Trained on mathematic and scientific problems, it is great for logic-based tasks. 

### JAIS 30b Chat

Usage: `llm -m github/jais-30b-chat`

**Publisher:** Core42 

**Description:** JAIS 30b Chat is an auto-regressive bilingual LLM for Arabic & English with state-of-the-art capabilities in Arabic. 

### Mistral Medium 3 (25.05)

Usage: `llm -m github/mistral-medium-2505`

**Publisher:** Mistral AI 

**Description:** Mistral Medium 3 is an advanced Large Language Model (LLM) with state-of-the-art reasoning, knowledge, coding and vision capabilities. 

### Mistral Small 3.1

Usage: `llm -m github/mistral-small-2503`

**Publisher:** Mistral AI 

**Description:** Enhanced Mistral Small 3 with multimodal capabilities and a 128k context length. 

### OpenAI o1

Usage: `llm -m github/o1`

**Publisher:** OpenAI 

**Description:** Focused on advanced reasoning and solving complex problems, including math and science tasks. Ideal for applications that require deep contextual understanding and agentic workflows. 

### OpenAI o1-mini

Usage: `llm -m github/o1-mini`

**Publisher:** OpenAI 

**Description:** Smaller, faster, and 80% cheaper than o1-preview, performs well at code generation and small context operations. 

### OpenAI o1-preview

Usage: `llm -m github/o1-preview`

**Publisher:** OpenAI 

**Description:** Focused on advanced reasoning and solving complex problems, including math and science tasks. Ideal for applications that require deep contextual understanding and agentic workflows. 

### OpenAI o3

Usage: `llm -m github/o3`

**Publisher:** OpenAI 

**Description:** o3 includes significant improvements on quality and safety while supporting the existing features of o1 and delivering comparable or better performance. 

### OpenAI o3-mini

Usage: `llm -m github/o3-mini`

**Publisher:** OpenAI 

**Description:** o3-mini includes the o1 features with significant cost-efficiencies for scenarios requiring high performance. 

### OpenAI o4-mini

Usage: `llm -m github/o4-mini`

**Publisher:** OpenAI 

**Description:** o4-mini includes significant improvements on quality and safety while supporting the existing features of o3-mini and delivering comparable or better performance. 

### OpenAI Text Embedding 3 (large)

Usage: `llm -m github/text-embedding-3-large`

**Publisher:** OpenAI 

**Description:** Text-embedding-3 series models are the latest and most capable embedding model from OpenAI. 

### OpenAI Text Embedding 3 (small)

Usage: `llm -m github/text-embedding-3-small`

**Publisher:** OpenAI 

**Description:** Text-embedding-3 series models are the latest and most capable embedding model from OpenAI. 

