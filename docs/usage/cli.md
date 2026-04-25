## REPL
```shell
lpy
#or
lpy -t #if you want to print python translation.
```

The REPL features Emacs-style keybindings (C-a, C-e, C-b, C-f, M-b, M-f, etc.), arrow key navigation, and persistent command history saved to `~/.lpy_history`.

### Auto-closing pairs

The REPL automatically inserts closing brackets and quotes when you type an opening one:

| You type | REPL inserts | Cursor position |
|----------|-------------|-----------------|
| `(`      | `()`        | between `(` and `)` |
| `[`      | `[]`        | between `[` and `]` |
| `{`      | `{}`        | between `{` and `}` |
| `'`      | `''`        | between the quotes |
| `"`      | `""`        | between the quotes |

- Typing a closing character (`)`, `]`, `}`) when the cursor is already before one will skip over it instead of inserting a duplicate.
- Typing a quote (`'`, `"`) when the cursor is before the same quote will skip over it.
- Pressing backspace between an empty pair deletes both characters.

## Run from source
```shell
lpy {filename}.lpy
```
## Run translation
```shell
l2py {filename}.lpy
```
It just displays translation. (don't run it)
## Run Tests (for development)
```shell
# in project root directory
pytest
#or
lpy -m pytest
```