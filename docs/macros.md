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