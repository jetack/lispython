# Changelog

All notable changes to lispython are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The nREPL `eval` op reads the standard `file`, `line` and `column` fields
  (where the code sits in the editor's buffer, 1-based). Tracebacks and
  compile errors then name that file with buffer line numbers and print the
  source line, including for definitions evaluated in earlier requests and
  for `load-file`.

### Changed

- An nREPL eval without `file` is named `<nrepl-N>` (a new N per eval)
  instead of `<nrepl>`, which can break a client that matches `<nrepl>` in
  error text.
- An invalid `line` or `column` (not an integer from 1 to
  `NREPL-MAX-POSITION-LINE`, 1,000,000, or `NREPL-MAX-POSITION-COLUMN`,
  100,000) gets an `error` reply with the request id instead of being
  accepted.

### Fixed

- Code a macro builds (from a quasiquote template) takes the position of the
  macro call, so a traceback or compile error in it points at that call
  instead of line 0; arguments written by the user keep their own positions.
- nREPL eval tracebacks no longer include the server's own `nrepl-eval`
  frame.
- The nREPL server buffers bytes and decodes each complete line, so a
  multi-byte UTF-8 character split across two socket reads no longer drops
  the connection. An unknown op, a malformed request or a failing handler
  (such as `eval` without `code`) now gets an `error` reply echoing the
  request id, and the connection keeps serving; a line that is not valid
  UTF-8 or JSON (or JSON that is not an object, or is nested too deeply to
  parse) gets an id-less `error` reply. A result that cannot be
  serialised is an `error` reply too, `(exit)` in an eval is reported as a
  `SystemExit` eval error, and any socket error ends the connection quietly.
  A request line over 4 MiB gets one id-less `error` reply and is dropped up
  to its next newline, so a client that never sends a newline cannot grow the
  server's memory; the connection keeps serving.
- `as->` expands to an expression (a tuple of `:=` assignments indexed with
  `-1`), so it works in expression position such as a `return` value or a
  call argument instead of emitting an invalid `do(...)` call.
- `as->` renames only references to the threading name: a keyword-argument
  name (`(f :x x)` was `f(__x_0=__x_0)`, now `f(x=__x_0)`) and the params
  and body of a `fn`/`lambda` that binds the same name are left alone (its
  defaults are still renamed), and `~x` inside a quasiquote now gets the
  threaded value.
- `as->` also renames a dotted reference (`x.real` becomes `__x_0.real`) and
  a reference inside an f-string format spec (`f"{1:{x}}"`).
- Threading a literal into a call (`(-> 1 (f))`, `(->> 1 (f 2))`,
  `(-> "a" (f))`) compiles to `f(1)`, `f(2, 1)` and `f('a')` instead of
  failing with an `AttributeError`. A macro may put Python values (numbers,
  strings, bytes, `True`/`False`/`None`, lists) anywhere in the forms it
  returns; they become literals at the macro call's position.
- A string or bytes value a macro returns, a quoted string and a quoted
  symbol keep their exact value: a backslash, line ending, control or other
  non-printable character is escaped instead of being re-read as an escape
  (`(-> "a\\b" (f))` was `f('a\x08')`, `'a\b` was `Symbol('a\x08')`).
- A `for` or `while` with no target, iterable or test, or with no body
  before its `(else ...)` (`(for x in)`, `(for [x [1]] (print x))`,
  `(while x)`), is a compile error at the loop instead of an `IndexError`
  or a loop with an empty body that Python rejects.
- `^` followed by stars before a bracket or a quote (`^*(`, `^**[`, `^*'`,
  `^**~`) is a compile error at the prefix instead of a `KeyError`;
  `^*Ts` and `^**P` are unchanged.

### Changed

- A statement form (`do`, `=`, `if`, `return`, ...) in expression position is
  now a compile error instead of emitting an invalid call such as `do(...)`.
  A Python function named `do`, `deco`, `async_def`, etc. can no longer be
  called with lispython call syntax in expression position; they were already
  unreachable as statements.
- A Python hard keyword in call position (e.g. `(not a b)`, which used to
  compile to `not (a, b)`, silently false, or `(else ...)`) is now a compile
  error.

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
- Comprehension unpacking (3.15+), including `{**d for d dicts}`, which was
  mistranslated as a dict display; unary plus is gated to 3.15+.
- Errors the CPython compiler raises on `.lpy` modules, and lispython's own
  compile errors, are reported with the file, line and caret instead of a
  traceback by `lpy` (file, `-m`, `-c`), `l2py`, the REPL and nREPL eval.
- LSP: `textDocument/signatureHelp` and `workspace/symbol`; `(type NAME ...)`
  aliases are indexed as definitions.
- nREPL: completion responses carry type kinds, and the server writes
  `.nrepl-port` in the working directory for editor auto-discovery.

### Changed

- **The REPL and nREPL compile every statement of an input before running any
  of it**, and `lpy -c` compiles its whole input first. A compile error,
  lispython's or CPython's, now stops the input before any statement runs
  (macro expansion still happens first); a `SyntaxError` the program raises
  at run time keeps its own traceback.
- **Stricter parsing** rejects input that used to be accepted silently: an
  unclosed final form or mismatched closing bracket, an unterminated string,
  an f-string field holding more than one form, a lone `}` in an f-string,
  non-symbol macro names (`(defmacro m[T] ...)`), `(import "x")`,
  `(lazy-import "numpy")`, `(return a b)`, and `name[...]` glued under a
  prefix such as `^list[int]` in a parameter list. Malformed literals such as
  `09` or `\x4` are reported at their token.
- **Literal values**: `-1+2j` is now `(-1+2j)` rather than `-(1+2j)`; `\r\n`
  and lone `\r` become `\n` in f-strings too; a multi-line raw string no
  longer gains a literal `\n`; backslash-newline continues a string; a lone
  `"` inside `"""..."""` no longer ends the token.
- **F-strings follow Python's rules.** An uppercase `F` prefix (and `rf`,
  `fR`, ...) is a real f-string rather than being evaluated at compile time,
  `{x=}` echoes the expression text and defaults to `!r`, and format specs
  may nest replacement fields (`{x:>{w}.{p}}`). Unquotes expand inside
  f-string and t-string fields.
- **`(require mod [])` registers no macros**, like an empty import list; only
  an absent list (`(require mod)`) registers all of them. The LSP follows the
  same rule in go-to-definition and hover.
- `lpy -m` puts the current directory on `sys.path` like `python -m`,
  honouring `PYTHONSAFEPATH`.
- Requires `pygls` 2.0 or later.

### Fixed

- `(return [])` compiled to a bare `return` (yielding `None`); it now returns
  the empty list.
- `(case {})` accepts an empty mapping pattern.
- Signed literal match patterns: `(case -1 ...)` compiled to `case _(1):`.
- `(from . [a])` compiles; it raised `IndexError`.
- Quasiquoted f/t-string fields keep their conversion, spec and `=` text
  (`f"{x!r:>6}"` used to come back as `{x}`), and `f"{~x}"` with a literal
  macro argument no longer crashes.
- `l2py` emits valid Python for t-string interpolations.
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
