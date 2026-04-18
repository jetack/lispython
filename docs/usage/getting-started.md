# Getting Started
## Simple Web REPL
You can try LisPy without installing at
[https://jetack.github.io/lispy-web/](https://jetack.github.io/lispy-web/).
## Installation
### Using pip
```bash
pip install lispython
```
### Manual Installation (for development)
```bash
uv sync # install dependencies
uv pip install -e . # for development
```
#### uv
I recommend using [uv](https://docs.astral.sh/uv/) for development.
It manages the virtual environment and dependencies from `pyproject.toml` / `uv.lock` automatically.
```bash
uv run lpy # run commands inside the project environment
```