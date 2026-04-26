# LisPython
[![PyPI version](https://badge.fury.io/py/lispython.svg)](https://badge.fury.io/py/lispython)

LisPython is a Lisp-flavored syntax for Python with Lisp-style macros. Source files (`.lpy`) are transpiled to Python and executed on the standard CPython runtime.

## Documentation
Full documentation lives at [https://jetack.github.io/lispython/](https://jetack.github.io/lispython/).

## Quick Start
```lisp
(import math)

(def area [r]
  (return (* math.pi (** r 2))))

(print (area 3))
```
```bash
lpy example.lpy
```

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

## How to Run LisPython code
### Run from source
```bash
lpy {filename}.lpy
```

### Run REPL
```bash
lpy
# or
lpy -t # if you want to print python translation.
```

### Show translation
```bash
l2py {filename}.lpy
```
Prints the translated Python to stdout. It does not execute the code.

### Run Tests
```bash
# in project root directory
pytest
# or
lpy -m pytest
```

## LSP Server
LisPython ships with a language server (`lpy-lsp`) that speaks LSP over stdio. It provides:

- Diagnostics (parse / compile errors)
- Completions (special forms, builtins, file/workspace symbols)
- Hover documentation for special forms and builtins
- Document symbols
- Go-to-definition, including across `.lpy` files in the workspace

## nREPL Server
LisPython includes an nREPL server for REPL-driven development:

```bash
lpy --nrepl         # start on a random port
lpy --nrepl 7888    # start on a specific port
```

The server accepts newline-delimited JSON over TCP and supports:
- `eval` — evaluate LisPython code, return value/stdout/error
- `load-file` — load a `.lpy` file into the session
- `macroexpand` — expand macros and return the result
- `complete` — prefix and dot-completion from the live scope
- `docs` — signature and docstring for a symbol
- `annotate` — type tag (function/class/module/macro)

### Editor setup

#### VS Code
Install the [LisPython](https://marketplace.visualstudio.com/items?itemName=jetack.vscode-lispython) extension. It connects to both LSP and nREPL automatically.

#### Emacs
Use [`lpy-mode`](https://github.com/jetack/lpy-mode). It provides nREPL integration (eval, completion, eldoc, macroexpand) and LSP via eglot.

## Todo
### Python AST
- [ ] `type_comment` never considered. Later, it should be covered
- [ ] Any missing AST nodes in the version 3.12+
