# Macros
LisPy's macro system is highly inspired by Clojure's macro system.
You can check out the [Chapter 7](https://www.braveclojure.com/read-and-eval/) and [Chapter 8](https://www.braveclojure.com/writing-macros/) of "Brave Clojure" for more information about Clojure's macro system.
## Expression Nodes
These nodes are defined in `src/lispython/core/nodes.py`.

You can check out how to manipulate these nodes in

- the definitions of nodes themselves
- the definitions of built-in macros in `src/lispython/macros/sugar.lpy`.

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
(defmacro when [pred *body]
  (return `(if ~pred
             (do ~@body))))

(defmacro cond [*body]
  (def recur [*body]
    (if (< (len body) 4)
        (return `(if ~@body))
        (do (= [test then *orelse] body)
            (return `(if ~test ~then ~(recur *orelse))))))
  (return (recur *body)))

(defmacro -> [x *fs]
  (if (== 0 (len fs))
      (return x))
  (= [f *rest] fs)
  (if (isinstance f Paren)
      (do (f.list.insert 1 x)
          (return `(-> ~f ~@rest)))
      (return `(-> (~f ~x) ~@rest))))
```
You can find more in `src/lispython/macros/sugar.lpy`.

### `as->`
Unlike `->` (thread first) and `->>` (thread last), `as->` lets you name the threaded value and place it anywhere in each form:
```python
(as-> 0 x
  (+ x 10)     ;; x = 10
  (* 2 x)      ;; x = 20
  (str x))     ;; x = "20"
```

This expands to:
```python
x = 0
x = x + 10
x = 2 * x
x = str(x)
```

Useful when the threaded value doesn't always go in the first or last position.

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