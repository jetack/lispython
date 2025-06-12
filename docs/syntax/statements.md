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

## `raise`
LisPy version
```python
(raise Exception)
(raise Exception from ValueError)
```
Python version
```python
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