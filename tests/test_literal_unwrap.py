from lispy.tools import src_to_python_org

PREAMBLE = "(require lispy.macros *)\n"


class TestLiteralUnwrapInMacro:
    def test_number_is_int(self):
        src = PREAMBLE + """
(defmacro repeat-n [n *body]
  (= stmts [])
  (for _ in (range n)
    (stmts.extend body))
  (return `(do ~@stmts)))

(repeat-n 3 (print "hello"))
"""
        result = src_to_python_org(src)
        assert result.count("print('hello')") == 3

    def test_string_is_str(self):
        src = PREAMBLE + """
(defmacro greet [name]
  (= upper (.upper name))
  (return `(print ~upper)))

(greet "world")
"""
        result = src_to_python_org(src)
        assert "WORLD" in result

    def test_bool_is_bool(self):
        src = PREAMBLE + """
(defmacro if-flag [flag *body]
  (if flag
      (return `(do ~@body))
      (return `(pass))))

(if-flag True (print "yes"))
"""
        result = src_to_python_org(src)
        assert "print('yes')" in result

    def test_none_is_none(self):
        src = PREAMBLE + """
(defmacro maybe-wrap [val]
  (if (is val None)
      (return `(print "nothing"))
      (return `(print ~val))))

(maybe-wrap None)
"""
        result = src_to_python_org(src)
        assert "nothing" in result

    def test_symbol_stays_as_node(self):
        src = PREAMBLE + """
(defmacro my-inc [x]
  (return `(+ ~x 1)))

(my-inc y)
"""
        result = src_to_python_org(src)
        assert "y + 1" in result

    def test_expression_stays_as_node(self):
        src = PREAMBLE + """
(defmacro my-twice [expr]
  (return `(+ ~expr ~expr)))

(my-twice (+ a b))
"""
        result = src_to_python_org(src)
        assert "a + b + (a + b)" in result
