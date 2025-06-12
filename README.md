# LisPython
[![PyPI version](https://badge.fury.io/py/lispython.svg)](https://badge.fury.io/py/lispython)

# Documentation
You can find the documentation at [https://jetack.github.io/lispython/](https://jetack.github.io/lispython/).

# Installation
## Manual Installation (for development)
```bash
poetry install --no-root # for dependency
pip install -e . # for development
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


# Todo
## Environment
- [ ] Test on more python versions
- [ ] Some IDE plugins like hy-mode and jedhy for better editing experience.
## Macro System
- [ ] `as->` macro for syntactic sugar
- [ ] `gensym` for avoiding name collision
## Python AST
- [ ] `type_comment` never considered. Later, it should be covered