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
poetry install --no-root # for dependency
pip install -e . # for development
```
#### Poetry
I recommend using [Poetry](https://python-poetry.org/) for development.
And turn off virtual environment creation in Poetry settings.
```bash
poetry config virtualenvs.create false
```