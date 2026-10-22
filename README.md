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
uv run pytest practice/p01_optimizers -rA   # check one problem, listing each test
uv run pytest practice/p01_optimizers --solution  # run the same tests against solution.py
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

Each folder has one stub file you edit, its tests, and `solution.py`. Read the stub's docstring
for the problem statement. Try it before opening `solution.py`.

| # | Folder | Stub | Topic |
|---|--------|------|-------|
| 00 | `practice/p00_smoke` | `smoke.py` | Gradient descent with autograd (smoke test) |
| 01 | `practice/p01_optimizers` | `optimizers.py` | SGD (momentum) and Adam/AdamW as `torch.optim.Optimizer` subclasses |
| 02 | `practice/p02_autograd` | `functions.py` | Custom `torch.autograd.Function`s, numerical stability, gradcheck |
| 03 | `practice/p03_decorators` | `decorators.py` | Decorators and context managers for ML code |
| 04 | `practice/p04_attention` | `attention.py` | Scaled dot-product attention, masks, multi-head attention |
| 05 | `practice/p05_transformer` | `transformer.py` | LayerNorm, transformer block, tiny GPT with generation |
| 06 | `practice/p06_modern_arch` | `modern.py` | RMSNorm, RoPE, SwiGLU, grouped-query attention, KV cache |
| 07 | `practice/p07_trainer` | `trainer.py` | Production trainer: config, callbacks, scheduler, checkpoint/resume |

Suggested order: 01 → 07. p03 is independent; do it whenever.
