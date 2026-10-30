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
- `(raise)` compiles to a bare `raise`, re-raising the exception being
  handled.

### Changed

- An nREPL eval without `file` is named `<nrepl-N>` (a new N per eval)
  instead of `<nrepl>`, which can break a client that matches `<nrepl>` in
  error text.
- An invalid `line` or `column` (not an integer from 1 to
  `NREPL-MAX-POSITION-LINE`, 1,000,000, or `NREPL-MAX-POSITION-COLUMN`,
  100,000) gets an `error` reply with the request id instead of being
  accepted.

### Fixed

- LSP hover and signature help for a statement form show the same usage
  line as the compiler's error messages (`(def name [args] body...)`,
  `(for target in iterable body...)`), and now cover the `except`,
  `except*`, `else`, `finally` and `case` clauses, augmented assignment
  (`+=` and friends), and `|` as an or-pattern. For `|`, hover and signature
  help show both the bitwise-OR operator and the or-pattern usage.
  Completion now also lists the augmented assignment operators and the
  `except`, `else` and `finally` clauses as special forms (`except*` and
  `case` were already offered as keywords).
  Hover text for `defmacro`, `lambda` and `fn` says `[args]`, matching `def`.
- A complex number a macro returns shows in macroexpand output as `1+2j`
  instead of `(1+2j)`, which read back as a call, not a number. A negative
  nan (alone or as a complex part) shows as `(- (float "nan"))` instead of
  losing its sign.
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
  `(-> "a" (f))`, `(-> ... (f))`) compiles to `f(1)`, `f(2, 1)`, `f('a')`
  and `f(...)` instead of failing with an `AttributeError`, and
  `(-> None (f))`, which compiled to `f()`, now gives `f(None)`. A macro may
  put Python values (`None`, booleans, numbers, `...`, str, bytes, and
  lists and tuples of them) anywhere in the forms it returns, including
  inside `*`, `**`, `^` and quote forms (`` `(list *~y) ``); each compiles
  to the same value (including `inf`, `nan`, `-0.0`, complex numbers with a
  signed zero, `inf` or `nan` part and ints of any size) at the macro
  call's position, even where `float` or `complex` is rebound or inside a
  match pattern. Converting nested lists does not recurse, so deep nesting
  does not hit `RecursionError` there. Any other value (a dict, a set, an
  object), or a form or list that contains itself (also one unquoted with
  `~`), is a compile error at the call naming the macro.
- A str a macro returns keeps its backslashes literally: `"a\\b"` was
  compiled as `"a\b"` (backspace), and a quote, line ending or control
  character is kept as written. The same holds for quoted strings and
  quoted symbols (`'a\b` was `Symbol('a\x08')`); bytes are supported.
- A `for` or `while` with no target, iterable or test, or with no body
  before its `(else ...)` (`(for x in)`, `(for [x [1]] (print x))`,
  `(while x)`), is a compile error at the loop instead of an `IndexError`
  or a loop with an empty body that Python rejects.
- `^` followed by stars before a bracket or a quote (`^*(`, `^**[`, `^*'`,
  `^**~`, also with a space after the quote) or alone (`^*`, `^**`) is a
  compile error at the prefix instead of a `KeyError`, an `AttributeError`
  or a call such as `f(*', a)`; `^*Ts` and `^**P` are unchanged.
- Every malformed statement form is a compile error naming the problem and
  the form's usage, at the form, clause or operand at fault, instead of a
  `ValueError`, `IndexError`, `AssertionError`, `TypeError` or an AST that
  Python rejects: `if`, `def`, `class`, `with`, `try` and its clauses,
  `match` and `case`, `deco`, `raise`, `assert`, `=`, augmented assignment,
  `del`, `global` and `nonlocal` with missing parts (`(if x)`, `(def f)`,
  `(with [a])`, `(try (f))`, `(match x (case 1))`, `(= x)`, `(global)`, ...),
  extra parts (`(raise a b)`, `(= x ^int 1 2)`) or a dangling `as`
  (`(with [a as] ...)`, `(except [E as] ...)`). A `try` clause out of order
  (except after else or finally, except mixed with except*, a form after
  the clauses) is reported at that clause instead of as "except is a Python
  keyword".
- An assignment, augmented assignment, `del`, `for` or `with ... as` target
  that Python cannot bind (`(= 1 2)`, `(= (f) 1)`, `(+= [] 1)`, `(del *a)`)
  is a compile error at the target with CPython's message ("cannot assign to
  literal"), and so is an annotation anywhere but right after a single
  target (`(= ^int 1)`, `(= a b ^int 1)`) or a def/class name that is not a
  name (`(def [] [] 1)`, `(class 1 [])`).
- An empty `()` is a compile error at it ("use (,) for an empty tuple")
  instead of an `IndexError`, wherever it appears.
- A quote, quasiquote or unquote-splice with no form right after it
  (`(f ' a)`, `(f *' a)`, `(f ~@ a)`, `(f ')`, `',a`) is a compile error at
  the quote instead of a symbol `'`, `f(*', a)`, an unquote of `@` or
  "unmatched ')'". `~` alone is still the invert operator.
- A malformed `match` case pattern is a compile error at the part at fault,
  quoting the usage of its kind (class, mapping or sequence pattern, `(| ...)`),
  instead of an `IndexError`, `ValueError` or `AttributeError` or an AST
  that means something else: a dangling `:attr` (`(P :x)`), a positional
  pattern after a keyword one (`(P :x 1 2)`, which compiled to `P(2, x=1)`),
  a mapping key without a pattern (`{"a"}`), `**rest` not last (`{**r 1}`),
  `**_` (the rest must be a capture name), a key that is no literal or
  dotted name (`{a 1}`), a second `*rest` (`[*a *b]`), `*rest` outside a
  sequence (`(case *a ...)`), `(| 1)`, a class pattern head that is no name
  (`(1)`, `((f) 1)`), a `()` that is no call (`(+ 1 2)`, `(, a b)`,
  `f"x"`), `...` and a `:x` keyword token as a pattern.
- A name after `*`, `**` or `:` in a case pattern that is no Python
  identifier (`[*1a]`, `{**1a}`, `(P :1a 2)`) is a compile error instead of
  an AST naming `1a`.
- A malformed dotted name in a case pattern (`.x`, `a..b`, `x.`), as a value
  pattern, a mapping key or a class pattern head, is a compile error instead
  of an attribute with an empty name.

### Changed

- A statement form (`do`, `=`, `if`, `return`, ...) in expression position is
  now a compile error instead of emitting an invalid call such as `do(...)`.
  A Python function named `do`, `deco`, `async_def`, etc. can no longer be
  called with lispython call syntax in expression position; they were already
  unreachable as statements.
- A Python hard keyword in call position (e.g. `(not a b)`, which used to
  compile to `not (a, b)`, silently false, or `(else ...)`) is now a compile
  error.
- Statement forms that compiled while dropping part of what was written are
  now compile errors: `(deco d (f))`, whose definition is not a `def`,
  `async-def`, `class` or `deco` (the decorator was silently dropped);
  `(assert a b c)` (the third operand was ignored); a form other than
  `(case ...)` inside `match` (it was read as a case); and def arguments or
  class bases written as anything but a `[...]` bracket (`(def f (x) ...)`
  compiled). A `case` with `as` after its guard (`(case 1 if g as y ...)`,
  which compiled `as y` into the body) is refused: `as` comes before `if`.
- A `;` ends a symbol or number, so a comment needs no space before it:
  `a;c` was the symbol `a;c`, now `a` followed by a comment.
- A Python hard keyword as a symbol name, or as a part of a dotted symbol,
  is a compile error: `(raise from)`, `(= o.if 1)` (which compiled to an
  attribute named `if`), and a keyword as a def, class, global, nonlocal,
  `case ... as` or `except [... as]` name (`(def if [] 1)`,
  `(case 1 as as ...)`).
- `~` directly before a closing bracket, a comma, `;` or the end of input
  (`(f ~)`, `(f ~,a)`, `~;c`) is the symbol `~`, as before a space, instead
  of an unquote that took the next form or gave "unmatched ')'".
- A dotted name in a case pattern (`(case Color.RED ...)`) is a value
  pattern, compared with `==` as in Python. It compiled to a capture pattern
  that matched every subject, so code that relied on that now falls through
  to the next case.
- A `()` in a case pattern whose head is an expression form (`(sub x)`,
  `(ife a b c)`, `(is-not x)`, `(not-in x)`) is a compile error ("pattern
  expected") instead of a class pattern on a class of that name.
- A sign before anything but a number in a case pattern (`(- a)`, `{-a 1}`)
  reports "pattern can sign only a number literal, got (- a)" with the
  pattern usage, instead of "only a number literal can be signed in a match
  pattern".
- Deeply nested forms compile in time linear in their nesting: a nested list
  (`[[[...]]]`, also inside a macro call such as `when`) was exponential (20
  levels took seconds) and a form in head position (`((((f))))`) quadratic.
  Rendering a form and expanding macros no longer recurse, so a form nested
  past Python's recursion limit renders and expands, and compiling one is a
  positioned compile error, "form nested too deeply to compile", instead of
  a `RecursionError`; the same holds for an AST too deep for CPython's
  `compile()` (such as `(+ 1 1 ...)` with thousands of operands) or for
  `ast.unparse` in `l2py`.

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
