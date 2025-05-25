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

## Supported Models

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

### DeepSeek-V3

Usage: `llm -m github/DeepSeek-V3`

**Publisher:** DeepSeek 

**Description:** A strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

### DeepSeek-V3-0324

Usage: `llm -m github/DeepSeek-V3-0324`

**Publisher:** DeepSeek 

**Description:** DeepSeek-V3-0324 demonstrates notable improvements over its predecessor, DeepSeek-V3, in several key aspects, including enhanced reasoning, improved function calling, and superior code generation capabilities. 

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

### Mistral Large

Usage: `llm -m github/Mistral-large`

**Publisher:** Mistral AI 

**Description:** Mistral's flagship model that's ideal for complex tasks that require large reasoning capabilities or are highly specialized (Synthetic Text Generation, Code Generation, RAG, or Agents). 

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

### Phi-4-multimodal-instruct

Usage: `llm -m github/Phi-4-multimodal-instruct`

**Publisher:** Microsoft 

**Description:** First small multimodal model to have 3 modality inputs (text, audio, image), excelling in quality and efficiency 

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

### JAIS 30b Chat

Usage: `llm -m github/jais-30b-chat`

**Publisher:** Core42 

**Description:** JAIS 30b Chat is an auto-regressive bilingual LLM for Arabic & English with state-of-the-art capabilities in Arabic. 

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

### OpenAI o3-mini

Usage: `llm -m github/o3-mini`

**Publisher:** OpenAI 

**Description:** o3-mini includes the o1 features with significant cost-efficiencies for scenarios requiring high performance. 

### OpenAI Text Embedding 3 (large)

Usage: `llm -m github/text-embedding-3-large`

**Publisher:** OpenAI 

**Description:** Text-embedding-3 series models are the latest and most capable embedding model from OpenAI. 

### OpenAI Text Embedding 3 (small)

Usage: `llm -m github/text-embedding-3-small`

**Publisher:** OpenAI 

**Description:** Text-embedding-3 series models are the latest and most capable embedding model from OpenAI. 
