"""Check a built lispython release before it is published.

Only git-tracked files may ship: local builds under ignored directories
(e.g. a checkout in .claude/worktrees/) once leaked agent files and symlinks.
Usage: check_dist.py <dist-dir> <version>   (run from the repo root)
"""

import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    ".claude", ".agents", ".agent", ".worktrees", ".git",
    "__pycache__", ".nox", ".venv",
}  # fmt: skip
AGENT_FILES = {"AGENTS.md", "CLAUDE.md", "GEMINI.md"}


def path_problems(name: str) -> list[str]:
    path = PurePosixPath(name)
    problems = [f"forbidden directory {p!r}: {name}" for p in path.parts
                if p in FORBIDDEN_PARTS]  # fmt: skip
    if path.suffix in {".pyc", ".pyo"}:
        problems.append(f"bytecode file: {name}")
    if path.name in AGENT_FILES:
        problems.append(f"agent file: {name}")
    return problems


def check_sdist(path: Path, version: str, tracked: set[str]) -> list[str]:
    errors: list[str] = []
    root = f"lispython-{version}"
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not (m.isfile() or m.isdir()):
                errors.append(f"sdist: not a regular file or directory: {m.name}")
            errors += [f"sdist: {e}" for e in path_problems(m.name)]
            if m.isfile() and m.name != f"{root}/PKG-INFO":
                rel = m.name.removeprefix(root + "/")
                if rel not in tracked:
                    errors.append(f"sdist: untracked file: {m.name}")
        info = tar.extractfile(f"{root}/PKG-INFO")
        text = info.read().decode() if info else ""
        if f"\nVersion: {version}\n" not in "\n" + text:
            errors.append(f"sdist: PKG-INFO Version is not {version}")
    return errors


def check_wheel(path: Path, version: str, tracked: set[str]) -> list[str]:
    errors: list[str] = []
    dist_info = f"lispython-{version}.dist-info/"
    with zipfile.ZipFile(path) as whl:
        for i in whl.infolist():
            if stat.S_ISLNK(i.external_attr >> 16):
                errors.append(f"wheel: symlink: {i.filename}")
            errors += [f"wheel: {e}" for e in path_problems(i.filename)]
            if i.filename.startswith(dist_info) or i.is_dir():
                continue
            if i.filename.startswith("lispy/") and (f"src/{i.filename}" not in tracked):
                errors.append(f"wheel: untracked file: {i.filename}")
        try:
            meta = whl.read(f"{dist_info}METADATA").decode()
        except KeyError:
            meta = ""
            errors.append("wheel: METADATA missing")
        if f"\nVersion: {version}\n" not in "\n" + meta:
            errors.append(f"wheel: METADATA Version is not {version}")
    return errors


def main() -> int:
    dist, version = Path(sys.argv[1]), sys.argv[2]
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )
    sdist = f"lispython-{version}.tar.gz"
    wheel = f"lispython-{version}-py3-none-any.whl"
    found = sorted(p.name for p in dist.iterdir())
    errors: list[str] = []
    if found != sorted([sdist, wheel]):
        errors.append(f"dist must hold exactly {sdist} and {wheel}, found {found}")
    else:
        errors += check_sdist(dist / sdist, version, tracked)
        errors += check_wheel(dist / wheel, version, tracked)
    for e in errors:
        print(f"check_dist: {e}", file=sys.stderr)
    if not errors:
        print(f"check_dist: ok ({sdist}, {wheel})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
