"""The .lpy bytecode cache sees edits to every file a module's code came from.

Each import runs in a fresh interpreter, so the code comes from the on-disk
cache (under a private PYTHONPYCACHEPREFIX) whenever that cache is trusted.
"""

import json
import os
import subprocess
import sys
import time

import pytest

RUN = """
import importlib, json, os, sys
import lispy
from lispy.core.importer import lpy_cache_path
m = importlib.import_module(sys.argv[1])
cache = lpy_cache_path(m.__file__)
print(json.dumps({"result": getattr(m, "result", None),
                  "cached": os.path.exists(cache),
                  "mode": os.path.exists(cache) and os.stat(cache).st_mode}))
"""


@pytest.fixture
def modules(tmp_path):
    directory = tmp_path / "mods"
    directory.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "cache")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(directory), *filter(None, [env.get("PYTHONPATH")])]
    )

    def write(name, text):
        (directory / name).write_text(text)

    def run(module="a", before=""):
        proc = subprocess.run(
            [sys.executable, "-c", before + RUN, module],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        out["stderr"] = proc.stderr
        return out

    write.run = run
    write.dir = directory
    return write


def test_an_edit_to_a_required_macro_module_recompiles(modules):
    modules("b.lpy", "(defmacro val [] (return 1))")
    modules("a.lpy", "(require b *)\n(= result (val))")
    assert modules.run()["result"] == 1
    modules("b.lpy", "(defmacro val [] (return 222))")
    assert modules.run()["result"] == 222


def test_an_edit_two_requires_away_recompiles(modules):
    # a takes only outer, so c's macro is in a's table only through b's
    # record: b expanded inner when b compiled.
    modules("c.lpy", "(defmacro inner [] (return 1))")
    modules("b.lpy", "(require c *)\n(defmacro outer [] (return (inner)))")
    modules("a.lpy", "(require b [outer])\n(= result (outer))")
    assert modules.run()["result"] == 1
    modules("c.lpy", "(defmacro inner [] (return 333))")
    assert modules.run()["result"] == 333


def test_an_edit_to_a_module_a_macro_imports_as_it_expands_recompiles(modules):
    modules("h.py", "VALUE = 1\n")
    modules("b.lpy", "(defmacro val [] (import h) (return h.VALUE))")
    modules("a.lpy", "(require b *)\n(= result (val))")
    assert modules.run()["result"] == 1
    modules("h.py", "VALUE = 4444\n")
    assert modules.run()["result"] == 4444


def test_a_same_size_edit_within_one_second_recompiles(modules):
    second = (int(time.time()) - 10) * 10**9
    path = modules.dir / "a.lpy"
    modules("a.lpy", "(= result 1)")
    os.utime(path, ns=(second + 100, second + 100))
    assert modules.run()["result"] == 1
    modules("a.lpy", "(= result 2)")
    os.utime(path, ns=(second + 200, second + 200))
    assert modules.run()["result"] == 2


def test_unchanged_dependencies_load_the_cache(modules):
    modules(
        "b.lpy",
        '(defmacro val [] (import sys) (print "EXPANDING" :file sys.stderr)'
        " (return 1))",
    )
    modules("a.lpy", "(require b *)\n(= result (val))")
    first = modules.run()
    assert "EXPANDING" in first["stderr"]
    assert first["cached"]
    second = modules.run()
    assert second["result"] == 1
    assert "EXPANDING" not in second["stderr"]


def test_a_cached_module_a_macro_imports_brings_its_record(modules):
    # n is read only while m compiles (m's macro imports it as it expands),
    # so once m loads from its cache, n reaches a only through m's record.
    modules("n.py", "VALUE = 1\n")
    modules(
        "m.lpy",
        '(defmacro mv [] (import importlib) (return (. (importlib.import-module "n")'
        " VALUE)))\n(= VALUE (mv))",
    )
    modules(
        "a.lpy",
        '(defmacro val [] (import importlib) (return (. (importlib.import-module "m")'
        " VALUE)))\n(= result (val))",
    )
    modules.run("m")
    assert modules.run()["result"] == 1
    modules("n.py", "VALUE = 55555\n")
    assert modules.run()["result"] == 55555


PY_MACROS = "def val():\n    return {}\n\n\n__macro_namespace = {{'val': val}}\n"


def test_a_py_macro_module_is_stamped_as_it_was_read(modules):
    # b is edited after this process imported it, before a compiles: a's
    # code holds the old expansion, so its record must hold b's old stamp.
    modules("b.py", PY_MACROS.format(1))
    modules("a.lpy", "(require b *)\n(= result (val))")
    edit = (
        "import pathlib, lispy, b\n"
        f"pathlib.Path(b.__file__).write_text({PY_MACROS.format(666)!r})\n"
    )
    assert modules.run(before=edit)["result"] == 1
    assert modules.run()["result"] == 666


def test_the_cache_takes_the_umask(modules):
    modules("a.lpy", "(= result 1)")
    os.chmod(modules.dir / "a.lpy", 0o666)
    out = modules.run(before="import os\nos.umask(0o022)\n")
    assert out["cached"]
    assert out["mode"] & 0o022 == 0
