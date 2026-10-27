"""Adds `--solution`: run tests against the reference answers instead of your files.

- practice/<problem>/: one stub module (e.g. optimizers.py), its tests, and solution.py.
  With --solution, importing the stub module loads solution.py in its place.
- review/<sample>/: pr.py (the buggy pull request), its tests, and solution/fixed.py.
  With --solution, importing pr.py loads solution/fixed.py in its place.
- interview/<exercise>/: one module you write (e.g. rag.py), its tests, and
  solution/solution.py. With --solution, importing that module loads the solution instead.
"""

import importlib.abc
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


class _SolutionRedirect(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None) -> ModuleSpec | None:
        parts = fullname.split(".")
        if len(parts) != 3:
            return None
        top, folder, name = parts
        if top == "practice":
            if name == "solution" or name.startswith("test_") or name == "conftest":
                return None
            answer = ROOT / "practice" / folder / "solution.py"
        elif top == "review" and name == "pr":
            answer = ROOT / "review" / folder / "solution" / "fixed.py"
        elif top == "interview" and not name.startswith("test_") and name != "conftest":
            answer = ROOT / "interview" / folder / "solution" / "solution.py"
        else:
            return None
        if not answer.exists():
            return None
        return importlib.util.spec_from_file_location(fullname, answer)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--solution",
        action="store_true",
        help="Run tests against the reference answers (solution.py / fixed.py).",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--solution"):
        sys.meta_path.insert(0, _SolutionRedirect())


def pytest_report_header(config: pytest.Config) -> str:
    target = "reference answers" if config.getoption("--solution") else "your files"
    return f"practice target: {target}"
