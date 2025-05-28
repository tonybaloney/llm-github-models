# CONTRIBUTING

## Required tools

- [Python 3.9+](https://docs.python.org/3/using/index.html)

## Setup

1. Set up a virtual environment at `//.venv` and activate it
    (see [the docs](https://docs.python.org/3/library/venv.html) for more information)
1. `llm install -e '.[test]'` to install all dependencies

## Running tests

1. `pytest` to run tests

## Code formatting and type checks

Pull-requests will only pass in CI/CD if the following are met:

1. `ruff check`
2. `pyright llm_github_models.py`
3. `ruff format --check`

Run `ruff check --fix` to resort imports before submitting PRs, or commit another change. Run `ruff format` to bring the code file up to our style guidelines.
