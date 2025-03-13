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

### OpenAI GPT-4o

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/gpt-4o`

**Publisher:** OpenAI 

**Description:** gpt-4o offers a shift in how AI models interact with multimodal inputs. By seamlessly combining text, images, and audio, gpt-4o provides a richer, more engaging user experience.

Matching the intelligence of gpt-4 turbo, it is remarkably more efficient, delivering text at twice the speed and at half the cost. Additionally, GPT-4o exhibits the highest vision performance and excels in non-English languages compared to previous OpenAI models.

gpt-4o is engineered for speed and efficiency. Its advanced ability to handle complex queries with minimal resources can translate into cost savings and performance.

The introduction of gpt-4o opens numerous possibilities for businesses in various sectors:

1. **Enhanced customer service**: By integrating diverse data inputs, gpt-4o enables more dynamic and comprehensive customer support interactions.
2. **Advanced analytics**: Leverage gpt-4o's capability to process and analyze different types of data to enhance decision-making and uncover deeper insights.
3. **Content innovation**: Use gpt-4o's generative capabilities to create engaging and diverse content formats, catering to a broad range of consumer preferences.

#### Updates

`gpt-4o-2024-11-20`: this is the latest version of gpt-4o. Supports all previous output size (16,384) and features such as:  

- Text, image processing
- JSON Mode
- parallel function calling
- Enhanced accuracy and responsiveness
- Parity with English text and coding tasks compared to GPT-4 Turbo with Vision
- Superior performance in non-English languages and in vision tasks
- Support for enhancements
- Support for complex structured outputs.

#### Resources

- ["Hello gpt-4o" (OpenAI announcement)](https://openai.com/index/hello-gpt-4o/)
- [Introducing gpt-4o: OpenAI's new flagship multimodal model now in preview on Azure](https://azure.microsoft.com/blog/introducing-gpt-4o-openais-new-flagship-multimodal-model-now-in-preview-on-azure/)
 

### OpenAI GPT-4o mini

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/gpt-4o-mini`

**Publisher:** OpenAI 

**Description:** GPT-4o mini enables a broad range of tasks with its low cost and latency, such as applications that chain or parallelize multiple model calls (e.g., calling multiple APIs), pass a large volume of context to the model (e.g., full code base or conversation history), or interact with customers through fast, real-time text responses (e.g., customer support chatbots).

Today, GPT-4o mini supports text and vision in the API, with support for text, image, video and audio inputs and outputs coming in the future. The model has a context window of 128K tokens and knowledge up to October 2023. Thanks to the improved tokenizer shared with GPT-4o, handling non-English text is now even more cost effective.

GPT-4o mini surpasses GPT-3.5 Turbo and other small models on academic benchmarks across both textual intelligence and multimodal reasoning, and supports the same range of languages as GPT-4o. It also demonstrates strong performance in function calling, which can enable developers to build applications that fetch data or take actions with external systems, and improved long-context performance compared to GPT-3.5 Turbo.

#### Resources

- [OpenAI announcement](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 

### OpenAI o1

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/o1`

**Publisher:** OpenAI 

**Description:** #### o1 Series Models: Enhanced Reasoning and Problem Solving on Azure

The o1 series models are specifically designed to tackle reasoning and problem-solving tasks with increased focus and capability. These models spend more time processing and understanding the user's request, making them exceptionally strong in areas like science, coding, math and similar fields. For example, o1 can be used by healthcare researchers to annotate cell sequencing data, by physicists to generate complicated mathematical formulas needed for quantum optics, and by developers in all fields to build and execute multi-step workflows.

#### Key Capabilities of the o1 Series

- [New!] o1 adds advanced image analysis capabilities with the new version. Enhance your prompts and context with images for additional insights.
- Complex Code Generation: Capable of generating algorithms and handling advanced coding tasks to support developers.
- Advanced Problem Solving: Ideal for comprehensive brainstorming sessions and addressing multifaceted challenges.
- Complex Document Comparison: Perfect for analyzing contracts, case files, or legal documents to identify subtle differences.
- Instruction Following and Workflow Management: Particularly effective for managing workflows requiring shorter contexts.

#### Features and properties supported in o1 GA model

- Developer message replaces system message
- Reasoning effort as in `high`, `medium`, and `low`. It controls whether the model thinks "less" or "more" in terms of applying cognitive reasoning.
- Vision input, structured outputs, and tools.

#### Model Variants

- `o1`: The most capable model in the o1 series, offering enhanced reasoning abilities. Now generally available.
- `o1-mini`: A faster and more cost-efficient option in the o1 series, ideal for coding tasks requiring speed and lower resource consumption.

#### Limitations

o1 model does not include all the features available in other models.  

### OpenAI o1-mini

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/o1-mini`

**Publisher:** OpenAI 

**Description:** #### OpenAI's o1 Series Models: Enhanced Reasoning and Problem Solving on Azure

The OpenAI o1 series models are specifically designed to tackle reasoning and problem-solving tasks with increased focus and capability. These models spend more time processing and understanding the user's request, making them exceptionally strong in areas like science, coding, math and similar fields. For example, o1 can be used by healthcare researchers to annotate cell sequencing data, by physicists to generate complicated mathematical formulas needed for quantum optics, and by developers in all fields to build and execute multi-step workflows.

o1-mini is developed to provide a faster, cheaper reasoning model that is particularly effective at coding. As a smaller model, o1-mini is 80% cheaper than o1-preview, making it a powerful, cost-effective model for applications that require reasoning but not broad world knowledge.

Note: Configurable content filters are currently not available for o1-preview and o1-mini.

_IMPORTANT: o1-mini model is available for limited access. To try the model in the playground, registration is required, and access will be granted based on Microsoft’s eligibility criteria._

#### Key Capabilities of the o1 Series

- Complex Code Generation: Capable of generating algorithms and handling advanced coding tasks to support developers.
- Advanced Problem Solving: Ideal for comprehensive brainstorming sessions and addressing multifaceted challenges.
- Complex Document Comparison: Perfect for analyzing contracts, case files, or legal documents to identify subtle differences.
- Instruction Following and Workflow Management: Particularly effective for managing workflows requiring shorter contexts.

#### Model Variants

- o1-preview: The most capable model in the o1 series, offering enhanced reasoning abilities.
- o1-mini: A faster and more cost-efficient option in the o1 series, ideal for coding tasks requiring speed and lower resource consumption.

#### Limitations

o1-mini model is currently in preview and do not include some features available in other models, such as image understanding and structured outputs found in the GPT-4o and GPT-4o-mini models. For many tasks, the generally available GPT-4o models may still be more suitable.

#### Resources

- [OpenaI o1-mini model announcement](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/)
- [OpenAI o1-preview model announcement](https://openai.com/index/introducing-openai-o1-preview/)
- [Azure OpenAI blog announcement](https://aka.ms/new-models)
 

### OpenAI o1-preview

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/o1-preview`

**Publisher:** OpenAI 

**Description:** #### OpenAI's o1 Series Models: Enhanced Reasoning and Problem Solving on Azure

The OpenAI o1 series models are specifically designed to tackle reasoning and problem-solving tasks with increased focus and capability. These models spend more time processing and understanding the user's request, making them exceptionally strong in areas like science, coding, math and similar fields. For example, o1 can be used by healthcare researchers to annotate cell sequencing data, by physicists to generate complicated mathematical formulas needed for quantum optics, and by developers in all fields to build and execute multi-step workflows.

Note: Configurable content filters are currently not available for o1-preview and o1-mini.

_IMPORTANT: o1-preview model is available for limited access. To try the model in the playground, registration is required, and access will be granted based on Microsoft’s eligibility criteria._

#### Key Capabilities of the o1 Series

- Complex Code Generation: Capable of generating algorithms and handling advanced coding tasks to support developers.
- Advanced Problem Solving: Ideal for comprehensive brainstorming sessions and addressing multifaceted challenges.
- Complex Document Comparison: Perfect for analyzing contracts, case files, or legal documents to identify subtle differences.
- Instruction Following and Workflow Management: Particularly effective for managing workflows requiring shorter contexts.

#### Model Variants

- o1-preview: The most capable model in the o1 series, offering enhanced reasoning abilities.
- o1-mini: A faster and more cost-efficient option in the o1 series, ideal for coding tasks requiring speed and lower resource consumption.

#### Limitations

o1-preview model is currently in preview and do not include some features available in other models, such as image understanding and structured outputs found in the GPT-4o and GPT-4o-mini models. For many tasks, the generally available GPT-4o models may still be more suitable.

#### Resources

- [OpenaI o1-mini model announcement](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/)
- [OpenAI o1-preview model announcement](https://openai.com/index/introducing-openai-o1-preview/)
- [Azure OpenAI blog announcement](https://aka.ms/new-models)
 

### OpenAI o3-mini

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/o3-mini`

**Publisher:** OpenAI 

**Description:** #### o1 and o3 Series Models: Enhanced Reasoning and Problem Solving on Azure

The o1 and o3 series models are specifically designed to tackle reasoning and problem-solving tasks with increased focus and capability. These models spend more time processing and understanding the user's request, making them exceptionally strong in areas like science, coding, math and similar fields. For example, o1 can be used by healthcare researchers to annotate cell sequencing data, by physicists to generate complicated mathematical formulas needed for quantum optics, and by developers in all fields to build and execute multi-step workflows.

#### Key Capabilities of these models

- o1 added advanced image analysis capabilities with the new version. Enhance your prompts and context with images for additional insights.
- o3-mini follows o1 mini but adds the features supported by o1 like function calling and tools.
- Complex Code Generation: Capable of generating algorithms and handling advanced coding tasks to support developers.
- Advanced Problem Solving: Ideal for comprehensive brainstorming sessions and addressing multifaceted challenges.
- Complex Document Comparison: Perfect for analyzing contracts, case files, or legal documents to identify subtle differences.
- Instruction Following and Workflow Management: Particularly effective for managing workflows requiring shorter contexts.

#### Features and properties supported in o3-mini model

- Supports both System message and the new Developer message to improve upgrade experience.
- Reasoning effort as in `high`, `medium`, and `low`. It controls whether the model thinks "less" or "more" in terms of applying cognitive reasoning.
- Structured outputs and functions/tools.
- Context window: 200K, Max Completion Tokens: 100K

#### Model Variants

- `o3-mini`: Now includes the o1 features with significant cost-efficiencies for scenarios requiring high performance.
- `o1`: The most capable model in the o1 series, offering enhanced reasoning abilities. Now generally available.
- `o1-mini`: A faster and more cost-efficient option in the o1 series, ideal for coding tasks requiring speed and lower resource consumption.

#### Limitations

o1 model does not include all the features available in other models.  

### OpenAI Text Embedding 3 (large)

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/text-embedding-3-large`

**Publisher:** OpenAI 

**Description:** Text-embedding-3 series models are the latest and most capable embedding model. The text-embedding-3 models offer better average multi-language retrieval performance with the MIRACL benchmark while still maintaining performance for English tasks with the MTEB benchmark. 

### OpenAI Text Embedding 3 (small)

![Model Image](https://github.com//images/modules/marketplace/models/families/openai.svg)

Usage: `llm -m github/text-embedding-3-small`

**Publisher:** OpenAI 

**Description:** Text-embedding-3 series models are the latest and most capable embedding model. The text-embedding-3 models offer better average multi-language retrieval performance with the MIRACL benchmark while still maintaining performance for English tasks with the MTEB benchmark. 

### Phi-3.5-MoE instruct (128k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-5-MoE-instruct`

**Publisher:** Microsoft 

**Description:** Phi-3.5-MoE is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available documents - with a focus on very high-quality, reasoning dense data. The model supports multilingual and comes with 128K context length (in tokens). The model underwent a rigorous enhancement process, incorporating supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.

### Resources
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/phi3.5-techblog) <br>
📖 [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

### Model Architecture
Phi-3.5-MoE has 16x3.8B parameters with 6.6B active parameters when using 2 experts. The model is a mixture-of-expert decoder-only Transformer model using the tokenizer with vocabulary size of 32,064.

### Training Data
This is a static model trained on an offline dataset with 4.9T tokens and a cutoff date October 2023 for publicly available data. Future versions of the tuned models may be released as we improve models.
 

### Phi-3.5-mini instruct (128k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-5-mini-instruct`

**Publisher:** Microsoft 

**Description:** Phi-3.5-mini is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data. The model belongs to the Phi-3 model family and supports 128K token context length. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.

### Resources
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/phi3.5-techblog) <br>
📖 [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

### Model Architecture
Phi-3.5-mini has 3.8B parameters and is a dense decoder-only Transformer model using the same tokenizer as Phi-3 Mini. It is a text-only model best suited for prompts using chat format.

### Training Data
Phi-3.5-mini is a static model trained on an offline dataset with 3.4T tokens and a cutoff date October 2023 for publicly available data. Future versions of the tuned models may be released as we improve models. 

### Phi-3.5-vision instruct (128k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-5-vision-instruct`

**Publisher:** Microsoft 

**Description:** Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

### Resources
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/phi3.5-techblog) <br>
📖 [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

### Model Summary
|      |      |
|------|------|
| **Architecture** | Phi-3.5-vision has 4.2B parameters and contains image encoder, connector, projector, and Phi-3 Mini language model. |
| **Inputs** | Text and Image. It’s best suited for prompts using the chat format. |
| **Context length** | 128K tokens |
| **GPUs** | 256 A100-80G |
| **Training time** | 6 days |
| **Training data** | 500B tokens (vision tokens + text tokens) |
| **Outputs** | Generated text in response to the input |
| **Dates** | Trained between July and August 2024 |
| **Status** | This is a static model trained on an offline text dataset with cutoff date March 15, 2024. Future versions of the tuned models may be released as we improve models. |
| **Release date** | August 20, 2024 |
| **License** | MIT |
 

### Phi-3-medium instruct (128k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-medium-128k-instruct`

**Publisher:** Microsoft 

**Description:** The Phi-3-Medium-128K-Instruct is a 14B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties.
The model belongs to the Phi-3 family with the Medium version in two variants 4K and 128K which is the context length (in tokens) that it can support.

The model underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures.
When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3-Medium-128K-Instruct showcased a robust and state-of-the-art performance among models of the same-size and next-size-up.

#### Resources

🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

#### Model Architecture

Phi-3-Medium-128k-Instruct has 14B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

#### Training Datasets

Our training data includes a wide variety of sources, totaling 4.8 trillion tokens (including 10% multilingual), and is a combination of 
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
2) Newly created synthetic, "textbook - like" data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report). 

### Phi-3-medium instruct (4k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-medium-4k-instruct`

**Publisher:** Microsoft 

**Description:** The Phi-3-Medium-4K-Instruct is a 14B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties.
The model belongs to the Phi-3 family with the Medium version in two variants 4K and 128K which is the context length (in tokens) that it can support.

The model underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures.
When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3-Medium-4K-Instruct showcased a robust and state-of-the-art performance among models of the same-size and next-size-up.

#### Resources

🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

#### Model Architecture

Phi-3-Medium-4K-Instruct has 14B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

#### Training Datasets

Our training data includes a wide variety of sources, totaling 4.8 trillion tokens (including 10% multilingual), and is a combination of 
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
2) Newly created synthetic, "textbook-like" data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report). 

### Phi-3-mini instruct (128k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-mini-128k-instruct`

**Publisher:** Microsoft 

**Description:** The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets.
This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties.

After initial training, the model underwent a post-training process that involved supervised fine-tuning and direct preference optimization to enhance its ability to follow instructions and adhere to safety measures.
When evaluated against benchmarks that test common sense, language understanding, mathematics, coding, long-term context, and logical reasoning, the Phi-3 Mini-128K-Instruct demonstrated robust and state-of-the-art performance among models with fewer than 13 billion parameters.

#### Resources

🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

#### Model Architecture

Phi-3 Mini-128K-Instruct has 3.8B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

#### Training Datasets

Our training data includes a wide variety of sources, totaling 4.9 trillion tokens, and is a combination of 
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
2) Newly created synthetic, "textbook - like" data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report). 

### Phi-3-mini instruct (4k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-mini-4k-instruct`

**Publisher:** Microsoft 

**Description:** The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties.
The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support.

The model underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures.
When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3 Mini-4K-Instruct showcased a robust and state-of-the-art performance among models with less than 13 billion parameters.

#### Resources

🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

#### Model Architecture

Phi-3 Mini-4K-Instruct has 3.8B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

#### Training Datasets

Our training data includes a wide variety of sources, totaling 4.9 trillion tokens, and is a combination of 
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
2) Newly created synthetic, "textbook - like" data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report). 

### Phi-3-small instruct (128k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-small-128k-instruct`

**Publisher:** Microsoft 

**Description:** The Phi-3-Small-128K-Instruct is a 7B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model supports 128K context length (in tokens).

The model underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures.
When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3-Small-128K-Instruct showcased a robust and state-of-the-art performance among models of the same-size and next-size-up.

#### Resources

+ [Phi-3 Microsoft Blog](https://aka.ms/phi3blog-april)
+ [Phi-3 Technical Report](https://aka.ms/phi3-tech-report)

#### Model Architecture

Phi-3 Small-128K-Instruct has 7B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

#### Training Datasets

Our training data includes a wide variety of sources, totaling 4.8 trillion tokens (including 10% multilingual), and is a combination of 
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
2) Newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report). 

### Phi-3-small instruct (8k)

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-3-small-8k-instruct`

**Publisher:** Microsoft 

**Description:** The Phi-3-Small-8K-Instruct is a 7B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model supports 8K context length (in tokens).

The model underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures.
When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3-Small-8K-Instruct showcased a robust and state-of-the-art performance among models of the same-size and next-size-up.

#### Resources

🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>

#### Model Architecture

Phi-3 Small-8K-Instruct has 7B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.

#### Training Datasets

Our training data includes a wide variety of sources, totaling 4.8 trillion tokens (including 10% multilingual), and is a combination of 
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
2) Newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report). 

### Phi-4

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-4`

**Publisher:** Microsoft 

**Description:** Phi-4 is a state-of-the-art open model built upon a blend of synthetic datasets, data from filtered public domain websites, and acquired academic books and Q&A datasets. The goal of this approach was to ensure that small capable models were trained with data focused on high quality and advanced reasoning.

Phi-4 underwent a rigorous enhancement and alignment process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

For more information, reference the [Phi-4 Technical Report](https://www.microsoft.com/en-us/research/uploads/prod/2024/12/P4TechReport.pdf).

### Model Architecture

Phi-4 is a 14B parameters, dense decoder-only transformer model. 

### Training Data

Our training data is an extension of the data used for Phi-3 and includes a wide variety of sources from:

1. Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code.

2. Newly created synthetic, "textbook-like" data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.).

3. Acquired academic books and Q&A datasets.

4. High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

Multilingual data constitutes about 8% of our overall data. We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge.
 

### Phi-4-mini-instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-4-mini-instruct`

**Publisher:** Microsoft 

**Description:** Phi-4-mini-instruct is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data. The model belongs to the Phi-4 model family and supports 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning and direct preference optimization to support precise instruction adherence and robust safety measures.

Phi-4-mini-instruct is a dense decoder-only Transformer model with 3.8B parameters, offering key improvements over Phi-3.5-Mini, including a 200K vocabulary, grouped-query attention, and shared embedding. It is designed for chat-completion prompts, generating text based on user input, with a context length of 128K tokens. This static model was trained on an offline dataset with a June 2024 data cutoff. It supports many languages, including Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian.

The model is intended for broad multilingual commercial and research use. The model provides uses for general purpose AI systems and applications which require 1) memory/compute constrained environments; 2) latency bound scenarios; 3) strong reasoning (especially math and logic). The model is designed to accelerate research on language and multimodal models, for use as a building block for generative AI powered features. 

### Phi-4-multimodal-instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/microsoft.svg)

Usage: `llm -m github/Phi-4-multimodal-instruct`

**Publisher:** Microsoft 

**Description:** <!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

Phi-4-multimodal-instruct is a lightweight open multimodal foundation model that leverages the language, vision, and speech research and datasets used for Phi-3.5 and 4.0 models. The model processes text, image, and audio inputs, generating text outputs, and comes with 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning, and direct preference optimization to support precise instruction adherence and safety measures.

Phi-4-multimodal-instruct has 5.6B parameters and is a multimodal transformer model. The model has the pretrained Phi-4-mini as the backbone language model, and the advanced encoders and adapters of vision and speech. It has been trained on 5T text tokens, 2.3M speech hours, and 1.1T image-text tokens. This is a static model trained on offline datasets with the cutoff date of June 2024 for publicly available data. The supported languages for each modalities are:

- **Text**: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian
- **Image**: English
- **Audio**: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese 

### AI21 Jamba 1.5 Large

![Model Image](https://github.com//images/modules/marketplace/models/families/ai21 labs.svg)

Usage: `llm -m github/AI21-Jamba-1-5-Large`

**Publisher:** AI21 Labs 

**Description:** Jamba 1.5 Large is a state-of-the-art, hybrid SSM-Transformer instruction following foundation model. It's a Mixture-of-Expert model with 94B total parameters and 398B active parameters. The Jamba family of models are the most powerful & efficient long-context models on the market, offering a 256K context window, the longest available.. For long context input, they deliver up to 2.5X faster inference than leading models of comparable sizes. Jamba supports function calling/tool use, structured output (JSON), and grounded generation with citation mode and documents API. Jamba officially supports English, French, Spanish, Portuguese, German, Arabic and Hebrew, but can also work in many other languages.

**Model Developer Name**: Jamba 1.5 Large

#### Model Architecture

Jamba 1.5 Large is a state-of-the-art, hybrid SSM-Transformer instruction following foundation model

#### Model Variations	 

94B total parameters and 398B active parameters

#### Model Input
	
Models input text only.

#### Model Output

Models generate text only.

#### Model Dates

Jamba 1.5 Large was trained in Q3 2024 with data covering through early March 2024.

#### Model Information Table

| **Name**             | **Params**         | **Content Length**  |
|----------------------|--------------------|---------------------|
| **Jamba 1.5 Mini**   | 52B (12B active)   | 256K                |
| **Jamba 1.5 Large**  | 398B (94B active)  | 256K                | 

### AI21 Jamba 1.5 Mini

![Model Image](https://github.com//images/modules/marketplace/models/families/ai21 labs.svg)

Usage: `llm -m github/AI21-Jamba-1-5-Mini`

**Publisher:** AI21 Labs 

**Description:** Jamba 1.5 Mini is a state-of-the-art, hybrid SSM-Transformer instruction following foundation model. It's a Mixture-of-Expert model with 52B total parameters and 12B active parameters. The Jamba family of models are the most powerful & efficient long-context models on the market, offering a 256K context window, the longest available.. For long context input, they deliver up to 2.5X faster inference than leading models of comparable sizes. Jamba supports function calling/tool use, structured output (JSON), and grounded generation with citation mode and documents API. Jamba officially supports English, French, Spanish, Portuguese, German, Arabic and Hebrew, but can also work in many other languages.

**Model Developer Name**: AI21 Labs

#### Model Architecture
Jamba 1.5 Mini is a state-of-the-art, hybrid SSM-Transformer instruction following foundation model                                                                                                                             
#### Model Variations	
52B total parameters and 12B active parameters

#### Model Input	
Model inputs text only.

#### Model Output	
Model generates text only.

#### Model Dates
Jamba 1.5 Mini was trained in Q3 2024 with data covering through early March 2024.

#### *​​​Model Information Table*

| **Name**             | **Params**         | **Content Length**  |
|----------------------|--------------------|---------------------|
| **Jamba 1.5 Mini**   | 52B (12B active)   | 256K                |
| **Jamba 1.5 Large**  | 398B (94B active)  | 256K                | 

### Codestral 25.01

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Codestral-2501`

**Publisher:** Mistral AI 

**Description:** 
Codestral 25.01 is explicitly designed for code generation tasks. It helps developers write and interact with code through a shared instruction and completion API endpoint. As it masters code and can also converse in a variety of languages, it can be used to design advanced AI applications for software developers.

A model fluent in 80+ programming languages including Python, Java, C, C++, JavaScript, and Bash. It also performs well on more specific ones like Swift and Fortran. 
Improve developers productivity and reduce errors: it can complete coding functions, write tests, and complete any partial code using a fill-in-the-middle mechanism. 

**New standard on the performance/latency space with a 256k context window.**

#### Use-cases

- _Code generation_: code completion, suggestions, translation
- _Code understanding and documentation_: code summarization and explanation
- _Code quality_: code review, refactoring, bug fixing and test case generation
- _Code generation with fill-in-the-middle (FIM) completion_: users can define the starting point of the code using a prompt, and the ending point of the code using an optional suffix and an optional stop. The Codestral model will then generate the code that fits in between, making it ideal for tasks that require a specific piece of code to be generated.  

### Cohere Command R

![Model Image](https://github.com//images/modules/marketplace/models/families/cohere.svg)

Usage: `llm -m github/Cohere-command-r`

**Publisher:** Cohere 

**Description:** Command R is a highly performant generative large language model, optimized for a variety of use cases including reasoning, summarization, and question answering. 

The model is optimized to perform well in the following languages: English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Simplified Chinese, and Arabic.

Pre-training data additionally included the following 13 languages: Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, Persian.

#### Resources

For full details of this model, [release blog post](https://aka.ms/cohere-blog).

#### Model Architecture

This is an auto-regressive language model that uses an optimized transformer architecture. After pretraining, this model uses supervised fine-tuning (SFT) and preference training to align model behavior to human preferences for helpfulness and safety.

### Tool use capabilities

Command R has been specifically trained with conversational tool use capabilities. These have been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template will likely reduce performance, but we encourage experimentation.

Command R's tool use functionality takes a conversation as input (with an optional user-system preamble), along with a list of available tools. The model will then generate a json-formatted list of actions to execute on a subset of those tools. Command R may use one of its supplied tools more than once.

The model has been trained to recognise a special directly_answer tool, which it uses to indicate that it doesn't want to use any of its other tools. The ability to abstain from calling a specific tool can be useful in a range of situations, such as greeting a user, or asking clarifying questions. We recommend including the directly_answer tool, but it can be removed or renamed if required.

### Grounded Generation and RAG Capabilities

Command R has been specifically trained with grounded generation capabilities. This means that it can generate responses based on a list of supplied document snippets, and it will include grounding spans (citations) in its response indicating the source of the information. This can be used to enable behaviors such as grounded summarization and the final step of Retrieval Augmented Generation (RAG).This behavior has been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template may reduce performance, but we encourage experimentation.

Command R's grounded generation behavior takes a conversation as input (with an optional user-supplied system preamble, indicating task, context and desired output style), along with a list of retrieved document snippets. The document snippets should be chunks, rather than long documents, typically around 100-400 words per chunk. Document snippets consist of key-value pairs. The keys should be short descriptive strings, the values can be text or semi-structured.

By default, Command R will generate grounded responses by first predicting which documents are relevant, then predicting which ones it will cite, then generating an answer. Finally, it will then insert grounding spans into the answer. See below for an example. This is referred to as accurate grounded generation.

The model is trained with a number of other answering modes, which can be selected by prompt changes . A fast citation mode is supported in the tokenizer, which will directly generate an answer with grounding spans in it, without first writing the answer out in full. This sacrifices some grounding accuracy in favor of generating fewer tokens.

### Code Capabilities

Command R has been optimized to interact with your code, by requesting code snippets, code explanations, or code rewrites. It might not perform well out-of-the-box for pure code completion. For better performance, we also recommend using a low temperature (and even greedy decoding) for code-generation related instructions. 

### Cohere Command R 08-2024

![Model Image](https://github.com//images/modules/marketplace/models/families/cohere.svg)

Usage: `llm -m github/Cohere-command-r-08-2024`

**Publisher:** Cohere 

**Description:** Command R 08-2024  is a highly performant generative large language model, optimized for a variety of use cases including reasoning, summarization, and question answering. 

The model is optimized to perform well in the following languages: English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Simplified Chinese, and Arabic.

Pre-training data additionally included the following 13 languages: Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, Persian.

#### Model Architecture
This is an auto-regressive language model that uses an optimized transformer architecture. After pretraining, this model uses supervised fine-tuning (SFT) and preference training to align model behavior to human preferences for helpfulness and safety.

### Tool use capabilities
Command R 08-2024 has been specifically trained with conversational tool use capabilities. These have been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template will likely reduce performance, but we encourage experimentation.

Command R’s tool use functionality takes a conversation as input (with an optional user-system preamble), along with a list of available tools. The model will then generate a json-formatted list of actions to execute on a subset of those tools. Command R may use one of its supplied tools more than once.

The model has been trained to recognise a special directly_answer tool, which it uses to indicate that it doesn’t want to use any of its other tools. The ability to abstain from calling a specific tool can be useful in a range of situations, such as greeting a user, or asking clarifying questions. We recommend including the directly_answer tool, but it can be removed or renamed if required.

### Grounded Generation and RAG Capabilities

Command R 08-2024 has been specifically trained with grounded generation capabilities. This means that it can generate responses based on a list of supplied document snippets, and it will include grounding spans (citations) in its response indicating the source of the information. This can be used to enable behaviors such as grounded summarization and the final step of Retrieval Augmented Generation (RAG).This behavior has been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template may reduce performance, but we encourage experimentation.

Command R’s grounded generation behavior takes a conversation as input (with an optional user-supplied system preamble, indicating task, context and desired output style), along with a list of retrieved document snippets. The document snippets should be chunks, rather than long documents, typically around 100-400 words per chunk. Document snippets consist of key-value pairs. The keys should be short descriptive strings, the values can be text or semi-structured.

By default, Command R will generate grounded responses by first predicting which documents are relevant, then predicting which ones it will cite, then generating an answer. Finally, it will then insert grounding spans into the answer. See below for an example. This is referred to as accurate grounded generation.

The model is trained with a number of other answering modes, which can be selected by prompt changes . A fast citation mode is supported in the tokenizer, which will directly generate an answer with grounding spans in it, without first writing the answer out in full. This sacrifices some grounding accuracy in favor of generating fewer tokens.

### Code Capabilities
Command R 08-2024 has been optimized to interact with your code, by requesting code snippets, code explanations, or code rewrites. It might not perform well out-of-the-box for pure code completion. For better performance, we also recommend using a low temperature (and even greedy decoding) for code-generation related instructions.

### Structured Outputs
Structured Outputs ensures outputs from Cohere’s Command R 08-2024 model adheres to a user-defined response format. It supports JSON response format, including user-defined JSON schemas. This enables developers to reliably and consistently generate model outputs for programmatic usage and reliable function calls. Some examples include extracting data, formulating queries, and displaying model outputs in the UI. 

### Cohere Command R+

![Model Image](https://github.com//images/modules/marketplace/models/families/cohere.svg)

Usage: `llm -m github/Cohere-command-r-plus`

**Publisher:** Cohere 

**Description:** Command R+ is a highly performant generative large language model, optimized for a variety of use cases including reasoning, summarization, and question answering. 

The model is optimized to perform well in the following languages: English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Simplified Chinese, and Arabic.

Pre-training data additionally included the following 13 languages: Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, Persian.

#### Resources

For full details of this model, [release blog post](https://aka.ms/cohere-blog).

#### Model Architecture

This is an auto-regressive language model that uses an optimized transformer architecture. After pretraining, this model uses supervised fine-tuning (SFT) and preference training to align model behavior to human preferences for helpfulness and safety.

### Tool use capabilities

Command R+ has been specifically trained with conversational tool use capabilities. These have been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template will likely reduce performance, but we encourage experimentation.

Command R+'s tool use functionality takes a conversation as input (with an optional user-system preamble), along with a list of available tools. The model will then generate a json-formatted list of actions to execute on a subset of those tools. Command R+ may use one of its supplied tools more than once.

The model has been trained to recognise a special directly_answer tool, which it uses to indicate that it doesn't want to use any of its other tools. The ability to abstain from calling a specific tool can be useful in a range of situations, such as greeting a user, or asking clarifying questions. We recommend including the directly_answer tool, but it can be removed or renamed if required.

### Grounded Generation and RAG Capabilities

Command R+ has been specifically trained with grounded generation capabilities. This means that it can generate responses based on a list of supplied document snippets, and it will include grounding spans (citations) in its response indicating the source of the information. This can be used to enable behaviors such as grounded summarization and the final step of Retrieval Augmented Generation (RAG).This behavior has been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template may reduce performance, but we encourage experimentation.

Command R+'s grounded generation behavior takes a conversation as input (with an optional user-supplied system preamble, indicating task, context and desired output style), along with a list of retrieved document snippets. The document snippets should be chunks, rather than long documents, typically around 100-400 words per chunk. Document snippets consist of key-value pairs. The keys should be short descriptive strings, the values can be text or semi-structured.

By default, Command R+ will generate grounded responses by first predicting which documents are relevant, then predicting which ones it will cite, then generating an answer. Finally, it will then insert grounding spans into the answer. See below for an example. This is referred to as accurate grounded generation.

The model is trained with a number of other answering modes, which can be selected by prompt changes . A fast citation mode is supported in the tokenizer, which will directly generate an answer with grounding spans in it, without first writing the answer out in full. This sacrifices some grounding accuracy in favor of generating fewer tokens.

### Code Capabilities

Command R+ has been optimized to interact with your code, by requesting code snippets, code explanations, or code rewrites. It might not perform well out-of-the-box for pure code completion. For better performance, we also recommend using a low temperature (and even greedy decoding) for code-generation related instructions. 

### Cohere Command R+ 08-2024

![Model Image](https://github.com//images/modules/marketplace/models/families/cohere.svg)

Usage: `llm -m github/Cohere-command-r-plus-08-2024`

**Publisher:** Cohere 

**Description:** Command R+ 08-2024 is a highly performant generative large language model, optimized for a variety of use cases including reasoning, summarization, and question answering. 

The model is optimized to perform well in the following languages: English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Simplified Chinese, and Arabic.

Pre-training data additionally included the following 13 languages: Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, Persian.

#### Model Architecture
This is an auto-regressive language model that uses an optimized transformer architecture. After pretraining, this model uses supervised fine-tuning (SFT) and preference training to align model behavior to human preferences for helpfulness and safety.

### Tool use capabilities
Command R+ 08-2024 has been specifically trained with conversational tool use capabilities. These have been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template will likely reduce performance, but we encourage experimentation.

Command R+’s tool use functionality takes a conversation as input (with an optional user-system preamble), along with a list of available tools. The model will then generate a json-formatted list of actions to execute on a subset of those tools. Command R+ may use one of its supplied tools more than once.

The model has been trained to recognise a special directly_answer tool, which it uses to indicate that it doesn’t want to use any of its other tools. The ability to abstain from calling a specific tool can be useful in a range of situations, such as greeting a user, or asking clarifying questions. We recommend including the directly_answer tool, but it can be removed or renamed if required.

### Grounded Generation and RAG Capabilities

Command R+ 08-2024 has been specifically trained with grounded generation capabilities. This means that it can generate responses based on a list of supplied document snippets, and it will include grounding spans (citations) in its response indicating the source of the information. This can be used to enable behaviors such as grounded summarization and the final step of Retrieval Augmented Generation (RAG).This behavior has been trained into the model via a mixture of supervised fine-tuning and preference fine-tuning, using a specific prompt template. Deviating from this prompt template may reduce performance, but we encourage experimentation.

Command R+’s grounded generation behavior takes a conversation as input (with an optional user-supplied system preamble, indicating task, context and desired output style), along with a list of retrieved document snippets. The document snippets should be chunks, rather than long documents, typically around 100-400 words per chunk. Document snippets consist of key-value pairs. The keys should be short descriptive strings, the values can be text or semi-structured.

By default, Command R+ will generate grounded responses by first predicting which documents are relevant, then predicting which ones it will cite, then generating an answer. Finally, it will then insert grounding spans into the answer. See below for an example. This is referred to as accurate grounded generation.

The model is trained with a number of other answering modes, which can be selected by prompt changes . A fast citation mode is supported in the tokenizer, which will directly generate an answer with grounding spans in it, without first writing the answer out in full. This sacrifices some grounding accuracy in favor of generating fewer tokens.

### Code Capabilities
Command R+ 08-2024 has been optimized to interact with your code, by requesting code snippets, code explanations, or code rewrites. It might not perform well out-of-the-box for pure code completion. For better performance, we also recommend using a low temperature (and even greedy decoding) for code-generation related instructions.

### Structured Outputs
Structured Outputs ensures outputs from Cohere’s Command R+ 08-2024 model adheres to a user-defined response format. It supports JSON response format, including user-defined JSON schemas. This enables developers to reliably and consistently generate model outputs for programmatic usage and reliable function calls. Some examples include extracting data, formulating queries, and displaying model outputs in the UI. 

### Cohere Embed v3 English

![Model Image](https://github.com//images/modules/marketplace/models/families/cohere.svg)

Usage: `llm -m github/Cohere-embed-v3-english`

**Publisher:** Cohere 

**Description:** Cohere Embed English is the market’s leading multimodal (text, image) representation model used for semantic search, retrieval-augmented generation (RAG), classification, and clustering. Embed English has top performance on the HuggingFace MTEB benchmark and performs well on a variety of industries such as Finance, Legal, and General-Purpose Corpora.The model was trained on nearly 1B English training pairs. 

### Cohere Embed v3 Multilingual

![Model Image](https://github.com//images/modules/marketplace/models/families/cohere.svg)

Usage: `llm -m github/Cohere-embed-v3-multilingual`

**Publisher:** Cohere 

**Description:** Cohere Embed Multilingual is the market’s leading multimodal (text, image) representation model used for semantic search, retrieval-augmented generation (RAG), classification, and clustering. Embed Multilingual supports 100+ languages and can be used to search within a language (e.g., search with a French query on French documents) and across languages (e.g., search with an English query on Chinese documents). This model was trained on nearly 1B English training pairs and nearly 0.5B Non-English training pairs from 100+ languages. 

### DeepSeek-R1

![Model Image](https://github.com//images/modules/marketplace/models/families/deepseek.svg)

Usage: `llm -m github/DeepSeek-R1`

**Publisher:** DeepSeek 

**Description:** *Learn more: \[[original model announcement](https://github.com/deepseek-ai/DeepSeek-R1/tree/main)\]*

DeepSeek-R1 excels at reasoning tasks using a step-by-step training process, such as language, scientific reasoning, and coding tasks. It features 671B total parameters with 37B active parameters, and 128k context length.

DeepSeek-R1 builds on the progress of earlier reasoning-focused models that improved performance by extending Chain-of-Thought (CoT) reasoning. DeepSeek-R1 takes things further by combining reinforcement learning (RL) with fine-tuning on carefully chosen datasets. It evolved from an earlier version, DeepSeek-R1-Zero, which relied solely on RL and showed strong reasoning skills but had issues like hard-to-read outputs and language inconsistencies. To address these limitations, DeepSeek-R1 incorporates a small amount of cold-start data and follows a refined training pipeline that blends reasoning-oriented RL with supervised fine-tuning on curated datasets, resulting in a model that achieves state-of-the-art performance on reasoning benchmarks.

### Usage Recommendations

We recommend adhering to the following configurations when utilizing the DeepSeek-R1 series models, including benchmarking, to achieve the expected performance:

- Avoid adding a system prompt; all instructions should be contained within the user prompt.
- For mathematical problems, it is advisable to include a directive in your prompt such as: "Please reason step by step, and put your final answer within \boxed{}."
- When evaluating model performance, it is recommended to conduct multiple tests and average the results. 

### DeepSeek-V3

![Model Image](https://github.com//images/modules/marketplace/models/families/deepseek.svg)

Usage: `llm -m github/DeepSeek-V3`

**Publisher:** DeepSeek 

**Description:** *Learn more: \[[original model announcement](https://github.com/deepseek-ai/DeepSeek-V3/tree/main)\]*

DeepSeek-V3 is a Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. 

DeepSeek-V3 was pre-train on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. 

### Llama-3.2-11B-Vision-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Llama-3-2-11B-Vision-Instruct`

**Publisher:** Meta 

**Description:** The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text \+ images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The models outperform many of the available open source and closed multimodal models on common industry benchmarks.

**Model Developer**: Meta

#### Model Architecture

Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. To support image recognition tasks, the Llama 3.2-Vision model uses a separately trained vision adapter that integrates with the pre-trained Llama 3.1 language model. The adapter consists of a series of cross-attention layers that feed image encoder representations into the core LLM.

|  | Training Data | Params | Input modalities | Output modalities | Context length | GQA | Data volume | Knowledge cutoff |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama 3.2-Vision  | (Image, text) pairs | 11B (10.6) | Text \+ Image | Text  | 128k | Yes | 6B (image, text) pairs | December 2023 |
| Llama 3.2-Vision | (Image, text) pairs | 90B (88.8) | Text \+ Image | Text  | 128k | Yes | 6B (image, text) pairs  | December 2023 |

**Supported Languages:** For text only tasks, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported. Llama 3.2 has been trained on a broader collection of languages than these 8 supported languages. Note for image+text applications, English is the only language supported. 

Developers may fine-tune Llama 3.2 models for languages beyond these supported languages, provided they comply with the Llama 3.2 Community License and the Acceptable Use Policy. Developers are always expected to ensure that their deployments, including those that involve additional languages, are completed safely and responsibly.

#### Training Data

**Overview:** Llama 3.2-Vision was pretrained on 6B image and text pairs. The instruction tuning data includes publicly available vision instruction datasets, as well as over 3M synthetically generated examples.

**Data Freshness:** The pretraining data has a cutoff of December 2023\.
 

### Llama-3.2-90B-Vision-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Llama-3-2-90B-Vision-Instruct`

**Publisher:** Meta 

**Description:** The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text \+ images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The models outperform many of the available open source and closed multimodal models on common industry benchmarks.

**Model Developer**: Meta

#### Model Architecture

Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. To support image recognition tasks, the Llama 3.2-Vision model uses a separately trained vision adapter that integrates with the pre-trained Llama 3.1 language model. The adapter consists of a series of cross-attention layers that feed image encoder representations into the core LLM.

|  | Training Data | Params | Input modalities | Output modalities | Context length | GQA | Data volume | Knowledge cutoff |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama 3.2-Vision  | (Image, text) pairs | 11B (10.6) | Text \+ Image | Text  | 128k\* | Yes | 6B (image, text) pairs | December 2023 |
| Llama 3.2-Vision | (Image, text) pairs | 90B (88.8) | Text \+ Image | Text  | 128k\* | Yes | 6B (image, text) pairs  | December 2023 |

\* Note: Serverless APIs on Azure AI currently only support 8K context length.

**Supported Languages:** For text only tasks, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported. Llama 3.2 has been trained on a broader collection of languages than these 8 supported languages. Note for image+text applications, English is the only language supported. 

Developers may fine-tune Llama 3.2 models for languages beyond these supported languages, provided they comply with the Llama 3.2 Community License and the Acceptable Use Policy. Developers are always expected to ensure that their deployments, including those that involve additional languages, are completed safely and responsibly.

#### Training Data

**Overview:** Llama 3.2-Vision was pretrained on 6B image and text pairs. The instruction tuning data includes publicly available vision instruction datasets, as well as over 3M synthetically generated examples.

**Data Freshness:** The pretraining data has a cutoff of December 2023\.
 

### Llama-3.3-70B-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Llama-3-3-70B-Instruct`

**Publisher:** Meta 

**Description:** The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.

**Built with Llama**

**Model Architecture:** Llama 3.3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

|  | Training Data | Params | Input modalities | Output modalities | Context length | GQA | Token count | Knowledge cutoff |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama 3.3 (text only)  | A new mix of publicly available online data. | 70B | Multilingual Text | Multilingual Text and code  | 128k | Yes | 15T+* | December 2023 |

*Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.
 

### Meta-Llama-3.1-405B-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Meta-Llama-3-1-405B-Instruct`

**Publisher:** Meta 

**Description:** The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned
generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on
common industry benchmarks.

#### Model Architecture

Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

#### Training Datasets

**Overview:** Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples.

**Data Freshness:** The pretraining data has a cutoff of December 2023. 

### Meta-Llama-3.1-70B-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Meta-Llama-3-1-70B-Instruct`

**Publisher:** Meta 

**Description:** The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned
generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on
common industry benchmarks.

#### Model Architecture

Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

#### Training Datasets

**Overview:** Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples.

**Data Freshness:** The pretraining data has a cutoff of December 2023. 

### Meta-Llama-3.1-8B-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Meta-Llama-3-1-8B-Instruct`

**Publisher:** Meta 

**Description:** The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned
generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on
common industry benchmarks.

#### Model Architecture

Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

#### Training Datasets

**Overview:** Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples.

**Data Freshness:** The pretraining data has a cutoff of December 2023. 

### Meta-Llama-3-70B-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Meta-Llama-3-70B-Instruct`

**Publisher:** Meta 

**Description:** Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks. Further, in developing these models, we took great care to optimize helpfulness and safety. 

#### Model Architecture

Llama 3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

#### Training Datasets

**Overview** Llama 3 was pretrained on over 15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 10M human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.

**Data Freshness** The pretraining data has a cutoff of March 2023 for the 8B and December 2023 for the 70B models respectively.  

### Meta-Llama-3-8B-Instruct

![Model Image](https://github.com//images/modules/marketplace/models/families/meta.svg)

Usage: `llm -m github/Meta-Llama-3-8B-Instruct`

**Publisher:** Meta 

**Description:** Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks. Further, in developing these models, we took great care to optimize helpfulness and safety. 

#### Model Architecture

Llama 3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

#### Training Datasets

**Overview** Llama 3 was pretrained on over 15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 10M human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.

**Data Freshness** The pretraining data has a cutoff of March 2023 for the 8B and December 2023 for the 70B models respectively.  

### Ministral 3B

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Ministral-3B`

**Publisher:** Mistral AI 

**Description:** **Ministral 3B** is a state-of-the-art Small Language Model (SLM) optimized for edge computing and on-device applications. As it is designed for low-latency and compute-efficient inference, it it also the perfect model for standard GenAI applications that have real-time requirements and high-volume.

**Number of Parameters:** 3,6 billions

Ministral 3B and Ministral 8B set a new frontier in knowledge, commonsense, reasoning, function-calling, and efficiency in the sub-10B category, and can be used or tuned to a variety of uses, from orchestrating agentic workflows to creating specialist task workers. Both models support up to 128k context length (currently 32k on vLLM) and Ministral 8B has a special interleaved sliding-window attention pattern for faster and memory-efficient inference.

#### Use cases

Our most innovative customers and partners have increasingly been asking for local, privacy-first inference for critical applications such as on-device translation, internet-less smart assistants, local analytics, and autonomous robotics. Les Ministraux were built to provide a compute-efficient and low-latency solution for these scenarios. From independent hobbyists to global manufacturing teams, les Ministraux deliver for a wide variety of use cases.

Used in conjunction with larger language models such as Mistral Large, les Ministraux are also efficient intermediaries for function-calling in multi-step agentic workflows. They can be tuned to handle input parsing, task routing, and calling APIs based on user intent across multiple contexts at extremely low latency and cost.

_Source: [Un Ministral, des Ministraux - Introducing the world’s best edge models.](https://mistral.ai/news/ministraux/)_
 

### Mistral Large 24.11

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Mistral-Large-2411`

**Publisher:** Mistral AI 

**Description:** Mistral Large 24.11 is an advanced Large Language Model (LLM) with state-of-the-art reasoning, knowledge and coding capabilities.

**NEW FEATURES.** 

- **SYSTEM PROMPTS**
- **BETTER PERFORMANCE ON LONG CONTEXT**
- **IMPROVED FUNCTION CALLING**

**Multi-lingual by design.** Dozens of languages supported, including English, French, German, Spanish, Italian, Chinese, Japanese, Korean, Portuguese, Dutch and Polish

**Proficient in coding.** Trained on 80+ coding languages such as Python, Java, C, C++, JavaScript, and Bash. Also trained on more specific languages such as Swift and Fortran

**Agent-centric.** Best-in-class agentic capabilities with native function calling and JSON outputting 

**Advanced Reasoning.** State-of-the-art mathematical and reasoning capabilities

*Context length:* 128K tokens

*Input:* Models input text only.

*Output:* Models generate text only. 

### Mistral Nemo

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Mistral-Nemo`

**Publisher:** Mistral AI 

**Description:** Mistral Nemo is a cutting-edge Language Model (LLM) boasting state-of-the-art reasoning, world knowledge, and coding capabilities within its size category.

**Jointly developed with Nvidia.** This collaboration has resulted in a powerful 12B model that pushes the boundaries of language understanding and generation.

**Multilingual proficiency.** Mistral Nemo is equipped with a new tokenizer, Tekken, designed for multilingual applications. It supports over 100 languages, including but not limited to English, French, German, Spanish, Italian, Chinese, Japanese, Korean, Portuguese, Dutch, Polish, and many more. Tekken has proven to be more efficient than the Llama 3 tokenizer in compressing text for approximately 85% of all languages, with significant improvements in Malayalam, Hindi, Arabic, and prevalent European languages.

**Agent-centric.** Mistral Nemo possesses top-tier agentic capabilities, including native function calling and JSON outputting.

**Advanced Reasoning.** Mistral Nemo demonstrates state-of-the-art mathematical and reasoning capabilities within its size category. 

### Mistral Large

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Mistral-large`

**Publisher:** Mistral AI 

**Description:** Mistral Large is Mistral AI's most advanced Large Language Model (LLM). It can be used on any language-based task thanks to its state-of-the-art reasoning and knowledge capabilities.

Additionally, Mistral Large is:

- **Specialized in RAG.** Crucial information is not lost in the middle of long context windows (up to 32K tokens).
- **Strong in coding.**  Code generation, review and comments. Supports all mainstream coding languages.
- **Multi-lingual by design.** Best-in-class performance in French, German, Spanish, and Italian - in addition to English. Dozens of other languages are supported.
- **Responsible AI.** Efficient guardrails baked in the model, with additional safety layer with safe_mode option

#### Resources

For full details of this model, please read [release blog post](https://aka.ms/mistral-blog). 

### Mistral Large (2407)

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Mistral-large-2407`

**Publisher:** Mistral AI 

**Description:** Mistral Large (2407) is an advanced Large Language Model (LLM) with state-of-the-art reasoning, knowledge and coding capabilities.

**Multi-lingual by design.** Dozens of languages supported, including English, French, German, Spanish, Italian, Chinese, Japanese, Korean, Portuguese, Dutch and Polish

**Proficient in coding.** Trained on 80+ coding languages such as Python, Java, C, C++, JavaScript, and Bash. Also trained on more specific languages such as Swift and Fortran

**Agent-centric.** Best-in-class agentic capabilities with native function calling and JSON outputting 

**Advanced Reasoning.** State-of-the-art mathematical and reasoning capabilities 

### Mistral Small

![Model Image](https://github.com//images/modules/marketplace/models/families/mistral ai.svg)

Usage: `llm -m github/Mistral-small`

**Publisher:** Mistral AI 

**Description:** Mistral Small is Mistral AI's most efficient Large Language Model (LLM). It can be used on any language-based task that requires high efficiency and low latency.

Mistral Small is:

- **A small model optimized for low latency.** Very efficient for high volume and low latency workloads. Mistral Small is Mistral's smallest proprietary model, it outperforms Mixtral 8x7B and has lower latency. 
- **Specialized in RAG.** Crucial information is not lost in the middle of long context windows (up to 32K tokens).
- **Strong in coding.** Code generation, review and comments. Supports all mainstream coding languages.
- **Multi-lingual by design.** Best-in-class performance in French, German, Spanish, and Italian - in addition to English. Dozens of other languages are supported.
- **Responsible AI.** Efficient guardrails baked in the model, with additional safety layer with safe_mode option

#### Resources

For full details of this model, please read [release blog post](https://aka.ms/mistral-blog). 

### JAIS 30b Chat

![Model Image](https://github.com//images/modules/marketplace/models/families/core42.svg)

Usage: `llm -m github/jais-30b-chat`

**Publisher:** Core42 

**Description:** JAIS 30b Chat from Core42 is an auto-regressive bi-lingual LLM for **Arabic** & **English** with state-of-the-art capabilities in Arabic.

#### Model Architecture

The model is based on transformer-based decoder-only (GPT-3) architecture and uses SwiGLU non-linearity. It uses LiBi position embeddings, enabling the model to extrapolate to long sequence lengths, providing improved context length handling. The tuned versions use supervised fine-tuning (SFT).

#### Training Datasets
 
**Overview:** The pretraining data for Jais-30b is a total of 1.63 T tokens consisting of English, Arabic, and code. Jais-30b-chat model is finetuned with both Arabic and English prompt-response pairs. We extended our finetuning datasets used for jais-13b-chat which included a wide range of instructional data across various domains. We cover a wide range of common tasks including question answering, code generation, and reasoning over textual content. To enhance performance in Arabic, we developed an in-house Arabic dataset as well as translating some open-source English instructions into Arabic.

**Data Freshness:** The pretraining data has a cutoff of December 2022, with some tuning data being more recent, up to October 2023. 

