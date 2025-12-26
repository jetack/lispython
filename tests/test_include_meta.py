from lispy.tools import l2py_s


def test_include_meta_flag_false():
    src = "(defmacro m [x] x)"
    out = l2py_s(src, include_meta=False)
    # When include_meta is False, the defmacro form itself should not appear in the output
    assert "defmacro" not in out and "def m" not in out


def test_include_meta_flag_true():
    src = "(defmacro m [x] x)"
    out = l2py_s(src, include_meta=True)
    # When include_meta is True, there should be some Python produced for the macro definition
    assert "def" in out or "__macro_namespace" in out


def test_include_meta_nested():
    src = "(list (defmacro m [x] x) 1)"
    out_false = l2py_s(src, include_meta=False)
    # Nested defmacro should not appear when include_meta is False
    assert "defmacro" not in out_false and "def m" not in out_false and "__macro_namespace" not in out_false

    out_true = l2py_s(src, include_meta=True)
    # With include_meta True the macro definition/import should be present
    assert "def" in out_true or "__macro_namespace" in out_true


def test_import_lispy_default_and_no_lispy():
    src = "(+ 1 2)"
    out_default = l2py_s(src)
    # By default we should prepend import lispy
    assert out_default.startswith("import lispy")

    out_no_lispy = l2py_s(src, no_lispy=True)
    # With no_lispy True we should not add the import
    assert not out_no_lispy.startswith("import lispy")
