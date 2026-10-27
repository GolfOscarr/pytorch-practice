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

## Code review practice (interview prep)

Each `review/rNN_*` folder is a pull request with planted bugs. You're the reviewer and the
bug fixer. The PR doesn't say how many bugs there are.

```
review/rNN_topic/
  pr.py              the pull request: read the description at the top, then review the code
  test_pr.py         run AFTER your review to see what you missed (tests don't name the bugs)
  solution/
    REVIEW.md        answer key: every bug (line, severity, why, fix) + production improvements
    fixed.py         corrected, production-quality version
```

Workflow:
1. Read `pr.py` and write your review (bugs, severity, fix, production suggestions).
   Don't run the tests yet.
2. Fix the bugs in `pr.py`.
3. `uv run pytest review/r01_transformer -rA` to see which behaviors are still broken.
4. Compare with `solution/REVIEW.md`. `git restore review/r01_transformer/pr.py` resets the PR.

`uv run pytest review/r01_transformer --solution` runs the tests against `solution/fixed.py`.

| # | Folder | Topic |
|---|--------|-------|
| 01 | `review/r01_transformer` | Transformer block / tiny GPT |
| 02 | `review/r02_training_loop` | Training loop: accumulation, clipping, scheduling, validation |
| 03 | `review/r03_ppo` | PPO objective and GAE |
| 04 | `review/r04_parallelism_config` | 3D parallelism planner for a GPU cluster |
| 05 | `review/r05_ddp` | DistributedDataParallel training script |
| 06 | `review/r06_data_pipeline` | Data pipeline: splits, leakage, workers, padding |
| 07 | `review/r07_mixed_precision` | Mixed precision and numerically safe losses |
| 08 | `review/r08_inference_server` | Batched generation and an async inference service |
| 09 | `review/r09_lora` | LoRA adapters |
| 10 | `review/r10_eval_metrics` | Streaming metrics, threshold selection, early stopping |
| 11 | `review/r11_contrastive` | CLIP-style contrastive (InfoNCE) loss |
| 12 | `review/r12_async_checkpoint` | Non-blocking checkpointing |

## Pair-programming practice (interview prep)

`interview/i01_mistral_rag`: build a RAG pipeline that combines a third-party retrieval API
(Wikipedia search) with the Mistral chat API, in timed stages, like a live pair-programming
round. Modeled on a reported Mistral AI Applied AI Engineer interview. Focus: API skills (auth,
timeouts, retries with backoff, response validation, streaming, async concurrency, testability).

```bash
cat interview/i01_mistral_rag/TASK.md                       # start here
uv run pytest interview/i01_mistral_rag -k stage1 -rA        # check one stage (offline, no key needed)
uv run python interview/i01_mistral_rag/run_live.py "question"   # live run (fake LLM if no MISTRAL_API_KEY)
cp interview/i01_mistral_rag/hard/rag.py interview/i01_mistral_rag/rag.py   # hard mode
```
After the session, read `solution/NOTES.md`: rubric, common mistakes, follow-up questions.
