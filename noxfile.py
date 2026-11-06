"""Test matrix over every maintained CPython: ``uvx nox -s tests``."""

import nox

nox.options.default_venv_backend = "uv"

PYTHONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]


@nox.session(python=PYTHONS)
def tests(session):
    # The lsp extra (basedpyright), so the real-backend tests run too.
    session.install(".[lsp]", "pytest")
    session.run("pytest", *session.posargs)


@nox.session
def docs(session):
    """Build the docs site strictly into a throwaway directory; never deploys."""
    session.install(".[docs]")
    site_dir = session.create_tmp() + "/site"
    session.run("mkdocs", "build", "--strict", "--site-dir", site_dir)
