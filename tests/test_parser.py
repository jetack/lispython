from lispy.core.nodes import Paren, Symbol
from lispy.core.parser import parse


class TestParser:
    def test_arrow_macro_definition(self):
        rst = parse("(defmacro -> [x *fs] x)")
        p = rst.pop()
        assert isinstance(p, Paren)
        assert str(p.list.__getitem__(1)) == "->"
        assert isinstance(p.list.__getitem__(1), Symbol)

    def test_arrow_macro_call(self):
        rst = parse("(-> 1 (+ 2))")
        p = rst.pop()
        assert isinstance(p, Paren)
        assert str(p.list.__getitem__(0)) == "->"
        assert isinstance(p.list.__getitem__(0), Symbol)

