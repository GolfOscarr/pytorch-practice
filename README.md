# pytorch-training

A practice environment for PyTorch on Apple Silicon (M4 Pro, MPS backend), managed with `uv`.

## Layout

```
practice/        your practice files go here (any test_*.py inside is picked up by pytest)
src/common/      shared helpers: get_device(), seed_everything()
tests/           environment tests
check_env.py     sanity check (torch, MPS, autograd, small training loop)
```

## Commands

```bash
uv run python practice/my_file.py    # run a file
uv run python check_env.py           # check the environment
uv run pytest                        # run tests
uv run pytest practice/test_adam.py  # run one test file
uv run ruff check . --fix            # lint and auto-fix
uv run ruff format .                 # format
uv run mypy practice                 # type-check
uv run ipython                       # REPL
uv add <package>                     # add a dependency
```

In any file, `from common import get_device, seed_everything` works.

## MPS notes

- A few ops aren't implemented on MPS. Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU.
- MPS doesn't support float64. Use float32.
- GPU calls are async. Call `torch.mps.synchronize()` before stopping a timer.

## VS Code

Open the folder (`code .`). The `.venv` interpreter, pytest discovery, Ruff format-on-save and a
"Python: Current File" debug config (F5) are preconfigured in `.vscode/`. If the interpreter isn't
picked up: Cmd+Shift+P → "Python: Select Interpreter" → `.venv/bin/python`.

## Problems

| # | Folder | Topic |
|---|--------|-------|
| 00 | `practice/p00_smoke` | Gradient descent with autograd (smoke test) |
