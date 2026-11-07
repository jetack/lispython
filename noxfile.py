"""Test matrix over every maintained CPython: ``uvx nox -s tests``."""

from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"

PYTHONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]


@nox.session(python=PYTHONS)
def tests(session):
    # The lsp extra (basedpyright, pyrefly), so the real-backend tests run too.
    session.install(".[lsp,dev]")
    session.run("pytest", *session.posargs)


@nox.session
def docs(session):
    """Build the docs site strictly into a throwaway directory; never deploys."""
    # Editable: a uv-cached wheel of the same version would hand mkdocs the old
    # form_docs and docstrings.
    session.install("-e", ".[docs]", "pytest")
    session.run(
        "python",
        "-c",
        "import lispy, pathlib, sys;"
        " here = pathlib.Path(sys.argv[1]).resolve() / 'src';"
        " print(lispy.__file__);"
        " sys.exit(here not in pathlib.Path(lispy.__file__).resolve().parents)",
        str(Path(__file__).parent),
    )
    site_dir = session.create_tmp() + "/site"
    # The hook tests also run in the tests session (dev extra has mkdocs); here
    # they run beside the build that uses the same hooks.
    session.run("pytest", "tests/test_docs_hooks.py")
    session.run("mkdocs", "build", "--strict", "--site-dir", site_dir)
