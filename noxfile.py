"""Test matrix over every maintained CPython: ``uvx nox -s tests``."""

from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"

PYTHONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]


# Both backends (basedpyright: lsp, pyrefly: lsp-pyrefly), so every
# real-backend test runs. The CI test job runs the same extras
# (tests/test_ci_workflow.py).
EXTRAS = ["dev", "lsp", "lsp-pyrefly"]


@nox.session(python=PYTHONS)
def tests(session):
    """The locked set (uv.lock), as `uv run --frozen` and CI use it."""
    flags = [arg for extra in EXTRAS for arg in ("--extra", extra)]
    session.run_install(
        "uv",
        "sync",
        "--frozen",
        "--python",
        session.virtualenv.location,
        *flags,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", *session.posargs)


@nox.session(python=PYTHONS)
def tests_latest(session):
    """The newest releases of every dependency, which the lock may lag."""
    session.install("." + "[" + ",".join(EXTRAS) + "]")
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
