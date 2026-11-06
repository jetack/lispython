# Macros
LisPy's macro system is highly inspired by Clojure's macro system.
You can check out the [Chapter 7](https://www.braveclojure.com/read-and-eval/) and [Chapter 8](https://www.braveclojure.com/writing-macros/) of "Brave Clojure" for more information about Clojure's macro system.
## Expression Nodes
These nodes are defined in `src/lispy/core/nodes.py`.

You can check out how to manipulate these nodes in

- the definitions of nodes themselves
- the definitions of built-in macros in `src/lispy/macros/sugar.lpy`.

## quote
```python
'expr
```
## syntax-quote
```python
`expr
```
## unquote
```python
~expr
```
## unquote-splicing
```python
~@expr
```
## Macro Definition
Just change `def` in function definition to `defmacro`. And macros usually return a quoted expression.
```python
(defmacro name [args*]
  body*)
```
### Macro Example
```python
(defmacro unless [test *body]
  (return `(if (not ~test)
             (do ~@body))))

(unless done
  (print "working"))
```
Python version
```python
if not done:
    print("working")
```
The built-in macros ([below](#built-in-macros)) are written the same way; their definitions are in `src/lispy/macros/sugar.lpy`.

## `require`
Loads the macros of a module so the file can use them. Macros are compile-time only: `require` leaves no runtime import behind.

LisPy version
```python
(require mod)          ; every macro, as mod.name
(require mod *)        ; every macro, unqualified
(require mod [a b])    ; only a and b, unqualified
(require mod [])       ; none
```
- `(require mod)` registers each macro of `mod` under the qualified name `mod.name`, so it is called as `(mod.name ...)`.
- `(require mod *)` registers every macro under its own name.
- `(require mod [a b])` registers just the listed macros under their own names.
- `(require mod [])` registers nothing.

A module that cannot be found is a warning (`could not load macros from 'mod' (module not found)`), not an error; a macro that was never registered then fails where it is used.

## Built-in macros
In a `.lpy` file the built-in macros need `(require lispy.macros *)` (or a selection such as `(require lispy.macros [-> when])`); the REPL and nREPL load them for you.

Every example below has `(require lispy.macros *)` at the top of the file; the Python version is what `l2py` prints.

### `cond`
Nested `if` statements: each test/then pair is one branch (`elif` after the first) and a trailing odd operand is the final `else`. It is a statement and yields no value: use `conde` for a value.

LisPy version
```python
(cond (< x 0) (print "neg")
      (== x 0) (print "zero")
      (print "pos"))
```
Python version
```python
if x < 0:
    print("neg")
elif x == 0:
    print("zero")
else:
    print("pos")
```
Note: a branch of several forms needs `(do ...)`.

### `conde`
The expression form of `cond`: a chain of `ife` whose trailing operand is the final else.

LisPy version
```python
(= label (conde (< x 0) "neg"
                (== x 0) "zero"
                "pos"))
```
Python version
```python
label = "neg" if x < 0 else "zero" if x == 0 else "pos"
```
Note: always give the final else. With an even number of operands the last test is never evaluated and its then becomes the else.

### `->`
Thread-first: the value becomes the first argument of the first form, that result the first argument of the next, and so on.

LisPy version
```python
(-> x (f 1) (g 2))
```
Python version
```python
g(f(x, 1), 2)
```
Note: a step that is not a parenthesised call, such as a bare name, is called with the value: `(-> x f (g 2))` is `g(f(x), 2)`.

### `->>`
Thread-last: the value becomes the last argument of the first form, that result the last argument of the next, and so on.

LisPy version
```python
(->> x (f 1) (g 2))
```
Python version
```python
g(2, f(1, x))
```
Note: a step that is not a parenthesised call, such as a bare name, is called with the value. After such a bare-name step the remaining steps thread first instead of last, so write every `->>` step as a parenthesised call.

### `as->`
Unlike `->` (thread first) and `->>` (thread last), `as->` lets you name the threaded value and place it anywhere in each form:
```python
(as-> 0 x
  (+ x 10)     ;; x = 10
  (* 2 x)      ;; x = 20
  (str x))     ;; x = "20"
```

This expands to a single expression, a tuple of walrus assignments indexed with `-1`:
```python
((__x_0 := 0), (__x_0 := (__x_0 + 10)), (__x_0 := (2 * __x_0)), str(__x_0))[-1]
```

Because it is an expression, `as->` works anywhere: as a return value, a call argument, a lambda body, or nested. CPython refuses `:=` in some positions (e.g. a comprehension's iterable, annotations, type-alias values); there it is a loud compile error, so bind the value first.

The name is replaced only where it is read as a variable, not in keyword-argument names, quoted data, or the body of a `fn`/`lambda` that rebinds it.

Useful when the threaded value doesn't always go in the first or last position.

### `fn`
Alias of `lambda`: an anonymous function whose body is one expression.

LisPy version
```python
(= double (fn [x] (* 2 x)))
```
Python version
```python
double = lambda x: 2 * x
```

### `comment`
Ignores its body: nothing in it is compiled or evaluated, and the form expands to `None`.

LisPy version
```python
(comment (print "never runs") (undefined-name))
(print "done")
```
Python version
```python
None
print("done")
```

### `when`
An `if` statement with no else, whose branch runs every body form in order. A statement: it yields no value.

LisPy version
```python
(when (> x 0)
  (print "positive")
  (= y x))
```
Python version
```python
if x > 0:
    print("positive")
    y = x
```

### `inc`
The expression `value + 1`; `value` itself is left unchanged.

LisPy version
```python
(= y (inc x))
```
Python version
```python
y = x + 1
```

### `dec`
The expression `value - 1`; `value` itself is left unchanged.

LisPy version
```python
(= z (dec x))
```
Python version
```python
z = x - 1
```

### `++`
The statement `target += 1`.

LisPy version
```python
(++ n)
```
Python version
```python
n += 1
```

### `--`
The statement `target -= 1`.

LisPy version
```python
(-- n)
```
Python version
```python
n -= 1
```

## gensym
`gensym` generates unique symbols to avoid variable name collisions in macros. It is automatically available inside `defmacro` bodies.

```python
(gensym)          ;; => Symbol("__gensym_0")
(gensym "tmp")    ;; => Symbol("__tmp_1")
(gensym 'tmp)     ;; => Symbol("__tmp_2")  (also accepts a quoted Symbol)
```

The prefix can be a string or a symbol. Each call increments a global counter, so every generated symbol is unique.

### Why gensym?
Without `gensym`, a macro that introduces a local variable can accidentally shadow a variable in the caller's scope:
```python
;; BAD: if the caller has a variable named `tmp`, this breaks
(defmacro broken-swap [a b]
  (return `(do (= tmp ~a)
               (= ~a ~b)
               (= ~b tmp))))
```

Using `gensym` prevents this:
```python
(defmacro swap [a b]
  (= tmp (gensym "tmp"))
  (return `(do (= ~tmp ~a)
               (= ~a ~b)
               (= ~b ~tmp))))
```

Now `(swap x y)` expands to something like:
```python
__tmp_0 = x
x = y
y = __tmp_0
```
The generated name `__tmp_0` won't collide with user variables.
