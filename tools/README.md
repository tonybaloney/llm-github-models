# Updating models

1. `python ./download_models_json.py`
1. `python ./parse_models_json.py`
1. Copy CHAT_MODELS and EMBEDDING_MODELS to `../llm_github_models.py`
1. Run `ruff format llm_github_models.py`
1. Move `models.fragment.md` to `../README.md`
