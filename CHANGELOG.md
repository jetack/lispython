# Changelog

All notable changes to lispython are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [0.5.0]

### Added

- Python 3.11 through 3.15 support, with each newer AST form gated on the
  running interpreter and a clear compile error on older ones. A `nox` matrix
  (`uvx nox -s tests`) runs the suite on every supported version.
- Type aliases: `(type NAME value)` compiles to a `type NAME = value`
  statement (3.12+). The three-operand `(type name bases ns)` call is unchanged.
- Generic definitions: `def`, `async def` and `class` names accept glued type
  parameters such as `f[T]`, including `ParamSpec`, `TypeVarTuple` and
  defaults (parameters 3.12+, defaults 3.13+).
- t-strings compile to `TemplateStr` (3.14+).
- `lazy-import` and `lazy-from` (3.15+).
- Comprehension unpacking (3.15+) and signed literal match patterns, with
  unary plus gated to 3.15+.
- Errors the CPython compiler raises on `.lpy` modules, and lispython's own
  compile errors, are reported with the file, line and caret instead of a
  traceback by `lpy` (file, `-m`, `-c`), `l2py`, the REPL and nREPL eval.
- LSP: `textDocument/signatureHelp` and `workspace/symbol`; `(type NAME ...)`
  aliases are indexed as definitions.
- nREPL: completion responses carry type kinds, and the server writes
  `.nrepl-port` in the working directory for editor auto-discovery.

### Changed

- **The REPL and nREPL compile every statement of an input before running any
  of it**, and `lpy -c` compiles its whole input first. A compile error now
  stops the input before any side effect; a `SyntaxError` the program raises
  at run time keeps its own traceback.
- **F-strings follow Python's rules.** An uppercase `F` prefix (and `rf`,
  `fR`, ...) is a real f-string rather than being evaluated at compile time,
  `{x=}` echoes the expression text and defaults to `!r`, and format specs
  may nest replacement fields (`{x:>{w}.{p}}`). Unquotes expand inside
  f-string and t-string fields.
- **`(require mod [])` registers no macros**, like an empty import list; only
  an absent list (`(require mod)`) registers all of them. The LSP follows the
  same rule in go-to-definition and hover.
- `lpy -m` puts the current directory on `sys.path` like `python -m`,
  honouring `-P` and `PYTHONSAFEPATH`.
- Requires `pygls` 2.0 or later.

### Fixed

- `(return [])` compiled to a bare `return` (yielding `None`); it now returns
  the empty list.
- `(case {})` accepts an empty mapping pattern.
- A lone starred subscript slice compiles to a one-element `Tuple`, like
  Python's `t[*s]`.
- LSP: workspace symbols no longer include installed packages or function
  locals, and report their real kinds; a buffer that is mid-edit keeps
  working; signature help is not offered inside a quoted form.
- nREPL: completion kinds treat scope bindings as shadowing builtins, never
  run user code to classify attributes, and report callables as functions;
  an unwritable `.nrepl-port` no longer stops the server.

## [0.4.7]

See the `v0.4.7` tag.
