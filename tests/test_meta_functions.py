import os.path as osp

from lispy.tools import l2py_s


class TestMetaFunctionsSync:
    """Test that core/meta_functions.py matches the transpiled result of core_meta_functions.lpy"""

    def test_meta_functions_in_sync(self):
        """core/meta_functions.py should match transpiled core_meta_functions.lpy"""
        base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        lpy_path = osp.join(base_dir, "src", "lispy", "core_meta_functions.lpy")
        py_path = osp.join(base_dir, "src", "lispy", "core", "meta_functions.py")

        with open(lpy_path, "r") as f:
            lpy_src = f.read()

        with open(py_path, "r") as f:
            py_contents = f.read()

        transpiled = l2py_s(lpy_src)

        assert transpiled == py_contents, (
            "core/meta_functions.py is out of sync with core_meta_functions.lpy. "
            "Run `lispy src/lispy/core_meta_functions.lpy` to regenerate."
        )
