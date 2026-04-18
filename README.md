# LisPython
[![PyPI version](https://badge.fury.io/py/lispython.svg)](https://badge.fury.io/py/lispython)

# Documentation
You can find the documentation at [https://jetack.github.io/lispython/](https://jetack.github.io/lispython/).

# Installation
## Manual Installation (for development)
```bash
uv sync # install dependencies
uv pip install -e . # for development
```
## Using pip
```bash
pip install lispython
```

# How to Run lispy code
## Run from source
```bash
lpy {filename}.lpy
```

## Run REPL
```bash
lpy
#or
lpy -t #if you want to print python translation.
```

## Run translation
```bash
l2py {filename}.lpy
```
It just displays translation. (don't run it)

## Run Tests
```bash
# in project root directory
python -m unittest
#or
lpy -m unittest
```

# LSP Server
LisPython ships with a language server (`lpy-lsp`) that speaks LSP over stdio. It provides:

- Diagnostics (parse / compile errors)
- Hover documentation for special forms
- Document symbols
- Go-to-definition, including across `.lpy` files in the workspace

## Editor setup
Point your editor's LSP client at the `lpy-lsp` command for files with the `.lpy` extension. The server speaks LSP over stdio.

### VSCode
Install the [LisPython](https://marketplace.visualstudio.com/) extension from the VSCode Marketplace.

### Emacs
Use [`lpy-mode`](https://github.com/jetack/lpy-mode) for syntax highlighting and LSP integration. For completion, install [`lpy-autocomplete`](https://github.com/jetack/lpy-autocomplete).


# Todo
## Environment
- [ ] Test on more python versions
- [ ] REPL should track history and arrow key navigation
- [ ] REPL multi-line input support
- [ ] Better compileation error messages
## Macro System
- [ ] `as->` macro for syntactic sugar
- [ ] `gensym` for avoiding name collision
## Python AST
- [ ] `type_comment` never considered. Later, it should be covered
- [ ] Any missing AST nodes in the version 3.12+