# .githooks

Tracked git hooks. They run only after a one-time activation per clone (see below).

## post-checkout

Provisions a new linked worktree (`git worktree add`) by symlinking the gitignored agent docs `AGENTS.md`, `CLAUDE.md` and `GEMINI.md` from the main checkout. It does nothing in the main checkout and only provisions files that are missing, so it is idempotent. Nothing is copied: the repo has no env file and no local database (`COPY_FILES` is empty).

The body is a verbatim copy of the canonical template at `~/vault/Vault/Projects/dotagents/guidelines/templates/post-checkout`. Only the `LINK_FILES`/`COPY_FILES` block varies per repo. Fix bugs in the template and re-copy; never patch this copy.

## Activation (once per clone)

Run in the main checkout:

```bash
git config core.hooksPath .githooks
```

⚠️ This replaces `.git/hooks`: any hook installed there (for example by an earlier `pre-commit install`) stops running.

This is local config and is not tracked. git looks for the hook relative to the root of the checkout you run `git worktree add` from, and runs the copy inside the new worktree. A worktree of a commit older than the hook's introduction has no `.githooks/post-checkout`, so it is not provisioned (and the catch-up command below is unavailable there).

A worktree created before activation can be provisioned from inside it:

```bash
.githooks/post-checkout 0 0 1
```

## pre-commit

With `core.hooksPath` set, git never reads `.git/hooks`, and `pre-commit install` refuses to run. The supported way to run the pre-commit framework is a tracked `.githooks/pre-commit` shim that calls `pre-commit run`, for example:

```bash
exec uv run --frozen --extra dev pre-commit run --hook-stage pre-commit
```

pre-commit is in the `dev` extra, so `uv run` needs `--extra dev` to provide it. (Or use `pre-commit` from the active venv.) Never run `pre-commit install`.

No such shim exists yet, so commits currently run no pre-commit checks.
