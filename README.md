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
- Hover documentation for special forms
- Document symbols
- Go-to-definition, including across `.lpy` files in the workspace

### Editor setup
Point your editor's LSP client at the `lpy-lsp` command for files with the `.lpy` extension. The server speaks LSP over stdio.

#### VSCode
Install the [LisPython](https://marketplace.visualstudio.com/items?itemName=jetack.vscode-lispython) extension from the VSCode Marketplace.

#### Emacs
Use [`lpy-mode`](https://github.com/jetack/lpy-mode) for syntax highlighting and LSP integration. For completion, install [`lpy-autocomplete`](https://github.com/jetack/lpy-autocomplete).

## Todo
### Environment
- [ ] Test on more python versions
- [ ] REPL should track history and arrow key navigation
- [ ] REPL multi-line input support
- [ ] Better compilation error messages
### Macro System
- [ ] `as->` macro for syntactic sugar
- [x] `gensym` for avoiding name collision
### Python AST
- [ ] `type_comment` never considered. Later, it should be covered
- [ ] Any missing AST nodes in the version 3.12+
