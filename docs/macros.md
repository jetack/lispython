---
render_macros: true
---

# Macros
LisPy's macro system is highly inspired by Clojure's macro system.
You can check out the [Chapter 7](https://www.braveclojure.com/read-and-eval/) and [Chapter 8](https://www.braveclojure.com/writing-macros/) of "Brave Clojure" for more information about Clojure's macro system.
## Expression Nodes
These nodes are defined in `src/lispy/core/nodes.py`.

You can check out how to manipulate these nodes in

- the definitions of nodes themselves
- the definitions of built-in macros in `src/lispy/macros/sugar.lpy`.

## quote
```lispython
'expr
```
## syntax-quote
```lispython
`expr
```
## unquote
```lispython
~expr
```
## unquote-splicing
```lispython
~@expr
```
## Macro Definition
Just change `def` in function definition to `defmacro`. And macros usually return a quoted expression.
```lispython
(defmacro name [args*]
  body*)
```
A `defmacro` body sees the node classes (`Paren`, `Bracket`, `Brace`, `Symbol`, `String`, `Constant`, `Keyword`, `Quote`, ... every class a quasiquote builds) and the helpers `gensym` and `replace_symbol` lexically: defining a macro binds none of them in the module, so a module's own `Paren` or `gensym` stays its own. A module-level helper function or quasiquote that needs them imports the same names with `(from lispy.core.builtins *)`.

### Macro Example
```lispython
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
```lispython
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
The built-in macros are loaded in every module and in the REPL with no `require`. A module's own `defmacro`s and explicit `require`s add to them, and `(require mod *)` never re-exports the built-ins.

To load only some of them, make `(require lispy.macros <selection>)` the module's first form (after an optional docstring); the selection replaces the built-ins:
```lispython
(require lispy.macros [-> when])   ; only -> and when
(require lispy.macros [])          ; none
(require lispy.macros *)           ; all (the default)
```
A bare `(require lispy.macros)` leaves only the qualified `(lispy.macros.inc ...)` names. A later `require` of `lispy.macros` only adds macros.

A macro is called by either spelling of its name, as every other name is: after `(defmacro my_mac ...)`, `(my-mac ...)` and `(my_mac ...)` both expand it, and a qualified `mod.my-mac` is `mod.my_mac`. Defining the other spelling (`(defmacro my-mac ...)` after `my_mac`) redefines that macro. A spelling that is not a hyphenated name (`->`, `ok?`, `x-`) is only itself.

Binding a name that is an active macro (`def`, `class`, `=`, parameters, a `defmacro`'s parameters and locals, loop and `with` targets, imports, match captures, ...) is a compile error, because `(inc ...)` would still expand the macro. Rename the binding, or opt out of that macro with a selection that omits it. A hyphenated name is one name with its Python name, so a binding `my_x` clashes with a macro `my-x` and a binding `my-x` with a macro `my_x`; any other spelling (`->`, `--`, `ok?`) clashes only with that same spelling, not with the name it mangles to (`_to_`, `__`, `ok_p`).

The Python version in each example is what `l2py` prints.

### `cond`
{{ macro_doc("cond") }}

LisPy version
```lispython
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

### `conde`
{{ macro_doc("conde") }}

LisPy version
```lispython
(= label (conde (< x 0) "neg"
                (== x 0) "zero"
                "pos"))
```
Python version
```python
label = "neg" if x < 0 else "zero" if x == 0 else "pos"
```

The final else is required: `(conde x 1 y 2)` is a compile error, as `1 if x else 2 if y` is in Python.

### `->`
{{ macro_doc("->") }}

LisPy version
```lispython
(-> x (f 1) (g 2))
```
Python version
```python
g(f(x, 1), 2)
```

### `->>`
{{ macro_doc("->>") }}

LisPy version
```lispython
(->> x (f 1) (g 2))
```
Python version
```python
g(2, f(1, x))
```

A bare name is called with the value and threading stays last: `(->> x f (g 1))` is `g(1, f(x))`.

### `as->`
{{ macro_doc("as->") }}

Unlike `->` (thread first) and `->>` (thread last), `as->` lets you name the threaded value and place it anywhere in each form:
```lispython
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
{{ macro_doc("fn") }}

LisPy version
```lispython
(= double (fn [x] (* 2 x)))
```
Python version
```python
double = lambda x: 2 * x
```

### `comment`
{{ macro_doc("comment") }}

LisPy version
```lispython
(comment (print "never runs") (undefined-name))
(print "done")
```
Python version
```python
None
print("done")
```

### `when`
{{ macro_doc("when") }}

LisPy version
```lispython
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
{{ macro_doc("inc") }}

LisPy version
```lispython
(= y (inc x))
```
Python version
```python
y = x + 1
```

### `dec`
{{ macro_doc("dec") }}

LisPy version
```lispython
(= z (dec x))
```
Python version
```python
z = x - 1
```

### `++`
{{ macro_doc("++") }}

LisPy version
```lispython
(++ n)
```
Python version
```python
n += 1
```

### `--`
{{ macro_doc("--") }}

LisPy version
```lispython
(-- n)
```
Python version
```python
n -= 1
```

## gensym
`gensym` generates unique symbols to avoid variable name collisions in macros. A `defmacro` body sees it without an import ([Macro Definition](#macro-definition)); elsewhere, import it with `(from lispy.core.builtins *)`.

```lispython
(gensym)          ;; => Symbol("__gensym_0")
(gensym "tmp")    ;; => Symbol("__tmp_1")
(gensym 'tmp)     ;; => Symbol("__tmp_2")  (also accepts a quoted Symbol)
```

The prefix can be a string or a symbol. Each call increments a global counter, so every generated symbol is unique.

### Why gensym?
Without `gensym`, a macro that introduces a local variable can accidentally shadow a variable in the caller's scope:
```lispython
;; BAD: if the caller has a variable named `tmp`, this breaks
(defmacro broken-swap [a b]
  (return `(do (= tmp ~a)
               (= ~a ~b)
               (= ~b tmp))))
```

Using `gensym` prevents this:
```lispython
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
