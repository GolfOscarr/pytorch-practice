"""Adds `--solution`: run a problem's tests against its solution.py instead of your stub.

Each problem folder has one stub module (e.g. practice/p01_optimizers/optimizers.py), its tests,
and solution.py. With --solution, importing the stub module loads solution.py in its place.
"""

import importlib.abc
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

PRACTICE_DIR = Path(__file__).parent / "practice"


class _SolutionRedirect(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None) -> ModuleSpec | None:
        parts = fullname.split(".")
        if len(parts) != 3 or parts[0] != "practice":
            return None
        name = parts[2]
        if name == "solution" or name.startswith("test_") or name == "conftest":
            return None
        solution = PRACTICE_DIR / parts[1] / "solution.py"
        if not solution.exists():
            return None
        return importlib.util.spec_from_file_location(fullname, solution)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--solution",
        action="store_true",
        help="Run practice tests against each problem's solution.py.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--solution"):
        sys.meta_path.insert(0, _SolutionRedirect())


def pytest_report_header(config: pytest.Config) -> str:
    target = "solution.py" if config.getoption("--solution") else "your stub files"
    return f"practice target: {target}"
