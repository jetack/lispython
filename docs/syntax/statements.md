# Statements
## Function Definitions
LisPy version
```python
(deco [decorator1
       decorator2]
  (def function-name [arg1 arg2 *args :kwarg1 value1 :kwarg2 value2 **kwargs]
    "docstring"
    body1
    body2
    (return value)))
```
Python version
```python
@decorator1
@decorator2
def function_name(arg1, arg2, *args, kwarg1=value1, kwarg2=value2, **kwargs):
    """docstring"""
    body1
    body2
    return value
```
### Async Function Definitions
Use `async-def` instead of `def`.

## Class Definition
LisPy version
```python
(deco [decorator1
       decorator2]
  (class class-name [base-class1 base-class2]
    "docstring"
    body))
```
Python version
```python
@decorator1
@decorator2
class class_name(base_class1, base_class2):
    """docstring"""

    body
```

## Generic Functions and Classes
Python 3.12+ ([PEP 695](https://peps.python.org/pep-0695/)); defaults need
Python 3.13+ ([PEP 696](https://peps.python.org/pep-0696/)). Write the type
parameters glued to the name, with no space: `f[T]` is a generic name, while
`f [T]` is a name followed by a list. The bracket uses the argument markers of
`def`: `^` for a bound (a tuple bound is a constraint list), `*` for a
TypeVarTuple, `**` for a ParamSpec and `:` for a default. On an older Python
these forms are a compile error naming the version they need.

A glued name is only valid as the name of `def`, `async-def`, `class` and
`type`; to subscript, use `(sub xs ...)`.

LisPy version
```python
(def first[T] [xs ^(sub list T)] ^T
  (return (sub xs 0)))

(class Box[T ^(, int str) *Ts **P] [Base]
  (pass))

(def f[:T str :U ^int bool :*Ts *(sub tuple int) :**P [int str]] []  ; Python 3.13+
  (pass))
```
Python version
```python
def first[T](xs: list[T]) -> T:
    return xs[0]

class Box[T: (int, str), *Ts, **P](Base):
    pass

def f[T = str, U: int = bool, *Ts = *tuple[int], **P = [int, str]]():  # Python 3.13+
    pass
```

## Type Alias
Python 3.12+ ([PEP 695](https://peps.python.org/pep-0695/)). As in Python,
`type` is a soft keyword: `(type NAME value)` and `(type NAME[T ...] value)`
are a type alias only as a statement with exactly two operands whose first is a
plain name. Every other shape is a call to the builtin `type`, so
`(= t (type x))` and `(type "N" (, object) {})` are unchanged. If you rebind
`type` and need a bare two-argument call as a statement, write
`(= _ (type a b))`. On an older Python the alias is a compile error naming the
version it needs.

LisPy version
```python
(type Number (| int float))
(type Vec[T] (sub list T))
```
Python version
```python
type Number = int | float
type Vec[T] = list[T]
```

## `del`
LisPy version
```python
(del x y z ...)
```
Python version
```python
del x, y, z, ...
```

## Assignment
LisPy version
```python
(= x value)
(= [x y *args] [value1 value2 value3 value4, ...]) ; destructuring assignment
```
Python version
```python
x = value
[x, y, *args] = [value1, value2, value3, value4, ...]
# then args = [value3, value4, ...]
```

## Augmented Assignment
LisPy version
```python
(+= x value)
(-= x value)
(*= x value)
(/= x value)
...
```
Python version
```python
x += value
x -= value
x *= value
x /= value
...
```

## Type Annotations
### Simple Type Annotations
LisPy version
```python
(= x ^type)
```
Python version
```python
x: type
```
### Assignment with Type Annotations
LisPy version
```python
(= a ^int 1)
```
Python version
```python
a: int = 1
```
### Function Definition with Type Annotations
LisPy version
```python
(def a [b ^int c] ^float
  (return (+ b c)))
```
Python version
```python
def a(b: int, c) -> float:
    return b + c
```
### Pydantic Example
LisPy version
```python
(class User [BaseModel]
  (= id ^int)
  (= name ^str "John Doe")
  (= signup_ts ^(| datetime None))
  (= tastes ^(sub dict (, str PositiveInt))))
```
Python version
```python
class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]
```

## `for`
LisPy version
```python
(for x in xs
  body1
  body2)
```
Python version
```python
for x in xs:
    body1
    body2
```

### Asynchronous `for`
Use `async-for` instead of `for`.

## `while`
LisPy version
```python
(while condition
  body1
  body2)
```
Python version
```python
while condition:
    body1
    body2
```

## `if`
LisPy version
```python
(if test
    then
    else)
```
Python version
```python
if test:
    then
else:
    else
```
## `do` block
If you want to use multiple statements in `then-body` or `else-body`, you can use `do` block.
LisPy version
```python
(if test
    (do then1 then2)
    (do else1 else2))
```
Python version
```python
if test:
    then1
    then2
else:
    else1
    else2
```

## `with`
### Simple `with`
LisPy version
```python
(with [expression]
  body)
```
Python version
```python
with expression:
    body
```
### `with` with alias
LisPy version
```python
(with [expression as target]
  body)
```
Python version
```python
with expression as target:
    body
```
### Asynchronous `with`
Use `async-with` instead of `with`.

## `match`
LisPy version
```python
(match x
  (case "Relevant"
        ...)
  (case None
        ...)
  (case [1 2]
        ...)
  (case [1 2 *rest]
        ...)
  (case [*_]
    ...)
  (case {1 _ 2 _}
    ...)
  (case {**rest}
    ...)
  (case (Point2D 0 0)
    ...)
  (case (Point3D :x 0 :y 0 :z 0)
    ...)
  (case [x] as y
    ...)
  (case _
    ...)
  (case (| [x] y)
    ...))
```
Python version
```python
match x:
    case "Relevant":
        ...
    case None:
        ...
    case [1, 2]:
        ...
    case [1, 2, *rest]:
        ...
    case [*_]:
        ...
    case {1: _, 2: _}:
        ...
    case {**rest}:
        ...
    case Point2D(0, 0):
        ...
    case Point3D(x=0, y=0, z=0):
        ...
    case [x] as y:
        ...
    case _:
        ...
    case [x] | y:
        ...
```
### Signed Literal Patterns
A negative number or a complex literal is a value pattern. A leading `+` needs
Python 3.15+; on an older Python it is a compile error naming the version.

LisPy version
```python
(match x
  (case -1
    ...)
  (case -1-2j
    ...)
  (case {-1 _}
    ...)
  (case +1  ; Python 3.15+
    ...))
```
Python version
```python
match x:
    case -1:
        ...
    case -1 - 2j:
        ...
    case {-1: _}:
        ...
    case +1:  # Python 3.15+
        ...
```

## `raise`
LisPy version
```python
(raise)
(raise Exception)
(raise Exception from ValueError)
```
Python version
```python
raise
raise Exception
raise Exception from ValueError
```

## `try`
LisPy version
```python
(try
  body1
  body2
  (except [Exception1]
    body3
    body4)
  (except [Exception2 as e]
    body5
    body6)
  (else
    body7
    body8)
  (finally
    body9
    body10))
```
Python version
```python
try:
    body1
    body2
except Exception1:
    body3
    body4
except Exception2 as e:
    body5
    body6
else:
    body7
    body8
finally:
    body9
    body10
```

### `except*` clause
LisPy version
```python
(try
  (raise BlockingIOError)
  (except* [BlockingIOError as e]
    (print (repr e))))
```
Python version
```python
try:
    raise BlockingIOError
except* BlockingIOError as e:
    print(repr(e))
```

## `assert`
LisPy version
```python
(assert condition)
(assert condition "message")
```
Python version
```python
assert condition
assert condition, "message"
```

## `import`
### Simple `import`
LisPy version
```python
(import module)
(import module as alias)
```
Python version
```python
import module
import module as alias
```
### `from` import
#### without alias
LisPy version
```python
(from module [name1 name2 name3])
```
Python version
```python
from module import name1, name2, name3
```
#### with alias
LisPy version
```python
(from module [name1
              name2 as alias
              name3])
```
Python version
```python
from module import name1
from module import name2 as alias
from module import name3
```
### Lazy imports
Python 3.15+ ([PEP 810](https://peps.python.org/pep-0810/)). `lazy-import` and
`lazy-from` take the same arguments as `import` and `from`; the module is
loaded when a name is first used. As in Python, `lazy-from` refuses `*` and
`__future__`, and a lazy import is not allowed inside a function, a class or a
`try` block. On an older Python these forms are a compile error naming the
version they need.

LisPy version
```python
(lazy-import json)
(lazy-import numpy as np)
(lazy-from json [dumps loads as load-json])
```
Python version
```python
lazy import json
lazy import numpy as np
lazy from json import dumps, loads as load_json
```

## `global`
LisPy version
```python
(global x y z ...)
```
Python version
```python
global x, y, z, ...
```
## `nonlocal`
LisPy version
```python
(nonlocal x y z ...)
```
Python version
```python
nonlocal x, y, z, ...
```
## `pass`
LisPy version
```python
(pass)
```
Python version
```python
pass
```
## `break`
LisPy version
```python
(break)
```
Python version
```python
break
```
## `continue`
LisPy version
```python
(continue)
```
Python version
```python
continue
```