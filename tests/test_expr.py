from lispy.tools import parse, src_to_python_org


class TestUnaryOpMethods:
    def test_unaryop(self):
        assert src_to_python_org("+2") == "+2"


class TestBinOpMethods:
    def test_binop(self):
        assert src_to_python_org("(+ 2 3)") == "2 + 3"


class TestBoolOpMethods:
    def test_boolop(self):
        assert src_to_python_org("(and 1 2)") == "1 and 2"


class TestCompareMethods:
    def test_eq(self):
        assert src_to_python_org("(== 1 2 3)") == "1 == 2 == 3"

    def test_noteq(self):
        assert src_to_python_org("(!= 1 2 3)") == "1 != 2 != 3"

    def test_lt(self):
        assert src_to_python_org("(< 1 2 3)") == "1 < 2 < 3"

    def test_lte(self):
        assert src_to_python_org("(<= 1 2 3)") == "1 <= 2 <= 3"

    def test_gt(self):
        assert src_to_python_org("(> 1 2 3)") == "1 > 2 > 3"

