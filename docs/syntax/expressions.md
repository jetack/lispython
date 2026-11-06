# Expressions
## Operators
### Boolean Operators
LisPy version
```lispython
(and expr1 expr2 ...)
```
Python version
```python
expr1 and expr2 and ...
```
### Binary Operators
LisPy version
```lispython
(+ n1 n2 ...)
```
Python version
```python
n1 + n2 + ...
```
### Unary Operators
LisPy version
```lispython
(- n)
```
Python version
```python
-n
```
#### `not`
LisPy version
```lispython
(not expr)
```
Python version
```python
not expr
```
#### `~`
LisPy version
```lispython
(~ n)
```
Python version
```python
~n
```
### Comparison Operators
#### `==`
LisPy version
```lispython
(== expr1 expr2 expr3 ...)
```
Python version
```python
expr1 == expr2 == expr3 == ...
```
#### `>`
LisPy version
```lispython
(> expr1 expr2 expr3 ...)
```
Python version
```python
expr1 > expr2 > expr3 > ...
```
#### `!=`
LisPy version
```lispython
(!= expr1 expr2 expr3 ...)
```
Python version
```python
expr1 != expr2 != expr3 != ...
```
#### `<`
LisPy version
```lispython
(< expr1 expr2 expr3 ...)
```
Python version
```python
expr1 < expr2 < expr3 < ...
```
#### `<=`
LisPy version
```lispython
(<= expr1 expr2 expr3 ...)
```
Python version
```python
expr1 <= expr2 <= expr3 <= ...
```
#### `>=`
LisPy version
```lispython
(>= expr1 expr2 expr3 ...)
```
Python version
```python
expr1 >= expr2 >= expr3 >= ...
```
#### `is`
LisPy version
```lispython
(is expr1 expr2 expr3 ...)
```
Python version
```python
expr1 is expr2 is expr3 is ...
```
#### `is-not`
LisPy version
```lispython
(is-not expr1 expr2 expr3 ...)
```
Python version
```python
expr1 is not expr2 is not expr3 is not ...
```
#### `in`
LisPy version
```lispython
(in expr1 expr2 expr3 ...)
```
Python version
```python
expr1 in expr2 in expr3 in ...
```
#### `not-in`
LisPy version
```lispython
(not-in expr1 expr2 expr3 ...)
```
Python version
```python
expr1 not in expr2 not in expr3 not in ...
```

## `:=`
LisPy version
```lispython
(:= name value)
```
Python version
```python
(name := value)
```

## Statements are not expressions
Statement forms (`do`, `=`, `+=`, `if`, `return`, `def`, ...) cannot appear inside an expression; the compiler reports an error. Use `(:= name value)` instead of `=`, `(ife test then else)` instead of `if`, and a tuple `(sub (, a b c) -1)` to sequence expressions.

## `lambda`
LisPy version
```lispython
(lambda [arg1 arg2 ...] body)
; or the built-in macro fn
(fn [arg1 arg2 ...] body)
```
Python version
```python
lambda arg1, arg2, ...: body
```
## `if` expression
LisPy version
```lispython
(ife test then other)
```
Python version
```python
then if test else other
```
## Collection Literals
### List
LisPy version
```lispython
[item1 item2 ...]
```
Python version
```python
[item1, item2, ...]
```
### Dictionary
LisPy version
```lispython
{key1 value1 key2 value2 ...}
```
Python version
```python
{key1: value1, key2: value2, ...}
```
### Set
LisPy version
```lispython
{, item1 item2 ...}
```
Python version
```python
{item1, item2, ...}
```
### Tuple
LisPy version
```lispython
(, item1 item2 ...)
```
Python version
```python
(item1, item2, ...)
```
## Comprehensions
### List Comprehension
LisPy version
```lispython
[(* 2 x) for x in (range 10) if (== (% x 2) 0)]
```
Python version
```python
[2 * x for x in range(10) if x % 2 == 0]
```
### Dictionary Comprehension
LisPy version
```lispython
{(* 2 x) (* 3 x) for x in (range 10) if (== (% x 2) 0)}
```
Python version
```python
{2 * x: 3 * x for x in range(10) if x % 2 == 0}
```
### Set Comprehension
LisPy version
```lispython
{, (* 2 x) for x in (range 10) if (== (% x 2) 0)}
```
Python version
```python
{2 * x for x in range(10) if x % 2 == 0}
```
### Generator Expression
LisPy version
```lispython
((* 2 x) for x in (range 10) if (== (% x 2) 0))
```
Python version
```python
(2 * x for x in range(10) if x % 2 == 0)
```
### Comprehension Unpacking
Python 3.15+ ([PEP 798](https://peps.python.org/pep-0798/)). On an older Python
these forms are a compile error naming the version they need.

LisPy version
```lispython
[*L for L in lists]
(*L for L in lists)
{, *s for s in sets}
{**d for d in dicts}
[*L async-for L in lists]
```
Python version
```python
[*L for L in lists]
(*L for L in lists)
{*s for s in sets}
{**d for d in dicts}
[*L async for L in lists]
```

## `await`
LisPy version
```lispython
(await expr)
```
Python version
```python
await expr
```
## `yield`
LisPy version
```lispython
(yield expr)
```
Python version
```python
yield expr
```
## `yield from`
LisPy version
```lispython
(yield-from expr)
```
Python version
```python
yield from expr
```
## Function Call
LisPy version
```lispython
(func-name arg1 arg2 *args :kwarg1 value1 :kwarg2 value2 **kwargs)
```
Python version
```python
func_name(arg1, arg2, *args, kwarg1=value1, kwarg2=value2, **kwargs)
```
## Formatted String Literals
LisPy version
```lispython
(= name "World")
f"Hello, {(* 2 name)}!"
```
Python version
```python
name = "World"
f"Hello, {2 * name}!"
```
Replacement fields follow Python's rules: `{{`/`}}` are literal braces,
`!r`/`!s`/`!a` conversions, format specs with nested fields, and `{x=}`.
A field is ordinary code, so a string in it is written with plain quotes as in
Python 3.12+ (`f"{(.get d "k")}"`), on every supported Python. The old
spelling with `\"` (`f"{(.get d \"k\")}"`) is deprecated: it still compiles,
with a `SyntaxWarning` (shown by default, as for an invalid escape sequence
in Python 3.12+), and a string opened with `\"` must be closed by
`\"`.
LisPy version
```lispython
f"{{literal}} {x!r:>{width}} {x=}"
```
Python version
```python
f"{{literal}} {x!r:>{width}} {x=}"
```

## Template String Literals
Python 3.14+ ([PEP 750](https://peps.python.org/pep-0750/)). The `t` prefix
combines with `r` like `f` does (`rt"..."`, `Tr"..."`) and builds a
`string.templatelib.Template`. Each interpolation's `expression` is the LisPy
text inside the braces. On an older Python a t-string is a compile error naming
the version it needs.

LisPy version
```lispython
(= name "World")
t"Hello, {(* 2 name)!r:>12}!"
```
Python version
```python
name = "World"
t"Hello, {2 * name!r:>12}!"  # interpolations[0].expression == "(* 2 name)"
```

## Attribute
Two forms read an attribute: a dotted name, and `(. obj attr)`.

### Dotted Name
A name with dots in it is Python's attribute access. It chains, and it works
anywhere a name does: as a value, a call head, an assignment or `del` target,
and in `import`.

LisPy version
```lispython
a.b.c
(obj.method arg :kwarg value)
(= obj.attr 1)
(+= obj.count 1)
(del obj.attr)
(import os.path)
```
Python version
```python
a.b.c
obj.method(arg, kwarg=value)
obj.attr = 1
obj.count += 1
del obj.attr
import os.path
```
Every part is an ordinary name, so hyphens become underscores (`a.b-c` is
`a.b_c`). The dotted name must be one token that starts with a name: `a.`
and `a..b` are errors, a lone `.a` as a value is an error (`(.a obj)` is the
method-call form below), and `a.b[0]` is refused (use `(sub a.b 0)`).

### `(. obj attr)`
The object may be any expression, which a dotted name cannot be: a literal,
call, subscription or other form on the left of the dot is not read as
attribute access (`(f).x` and `"s".upper` are errors). `(. obj attr)` takes
the attribute names after the object, so more than one chains them.

LisPy version
```lispython
(. obj attr)
(. obj a b)
(. (f) attr)
(. (sub items 0) attr)
(. "text" upper)
```
Python version
```python
obj.attr
obj.a.b
f().attr
items[0].attr
"text".upper
```
Each attribute must be a plain name, so `(. obj (method))` is an error: to
call a method on an expression use the method-call form below.

## Method Call
LisPy version
```lispython
(.method-name obj arg1 arg2 *args :kwarg1 value1 :kwarg2 value2 **kwargs)
```
Python version
```python
obj.method_name(arg1, arg2, *args, kwarg1=value1, kwarg2=value2, **kwargs)
```
## Subscription
### Simple Subscription
LisPy version
```lispython
(sub obj i1 i2)
```
Python version
```python
obj[i1][i2]
```
### Tuple Subscription
LisPy version
```lispython
(sub obj (, i1 i2 ...))
```
Python version
```python
obj[i1, i2, ...]
```
## Slice
### Slice Subscription
LisPy version
```lispython
(sub obj [: start stop step])
```
Python version
```python
obj[start:stop:step]
```
### Slice with Emptiness
LisPy version
```lispython
(sub obj [: stop])
(sub obj [: start _])
(sub obj [: start _ step])
```
Python version
```python
obj[:stop]
obj[start:]
obj[start::step]
```
