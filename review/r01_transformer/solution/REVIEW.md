# r01_transformer: answer key

PR: "Add a GPT-style decoder (pre-norm transformer block + tiny GPT LM)"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:141 | `F.cross_entropy(logits, targets)` on 3-D `(B, T, V)` logits: PyTorch treats dim 1 (`T`) as the class dim | crash (shape error whenever `targets` is passed, so training cannot run) |
| B2 | High | pr.py:155 | `generate` feeds the whole sequence back without cropping to `block_size` | crash (`ValueError` once prompt + generated > `block_size`) |
| B3 | Critical | pr.py:96 | `x + ln_1(attn(x))`: normalizes the sublayer *output*. That is neither pre-norm (`x + attn(ln(x))`) nor post-norm (`ln(x + attn(x))`) | silently wrong (unnormalized attention input; differs from the documented design) |
| B4 | Critical | pr.py:59 | `tril(..., diagonal=1)` lets token `t` attend to token `t+1` | silently wrong (future-token leak: training loss looks great, generation is garbage) |
| B5 | High | pr.py:71 | scores scaled by `sqrt(C)` (`n_embd`) instead of `sqrt(head_dim)` | silently wrong (softmax too flat, slower/worse training) |
| B6 | Critical | pr.py:76 | heads merged with `(B, H, T, D).reshape(B, T, C)` and no `transpose(1, 2)` first | silently wrong (mixes heads and positions, and leaks across time) |
| B7 | Medium | pr.py:113 | "tying" via `nn.Parameter(tok_emb.weight.clone())` makes a second, independent matrix | silently wrong (not tied: +V*C params, the two drift apart in training) |
| B8 | Medium | pr.py:144 | `generate` runs without `torch.no_grad()` and without `eval()` | perf (builds autograd graph every step) and reproducibility (dropout is active while decoding, so greedy is not deterministic) |

### B1: cross-entropy over the wrong dimension

`F.cross_entropy` expects input `(N, C)` or `(N, C, d1, ...)`: the class dimension is **dim 1**.
With `(B, T, V)` logits it reads `T` as the classes and expects targets of shape `(B, V)`, so it
raises an error (or, when `T == V`, quietly computes nonsense). Flatten the logits to `(B*T, V)` and the targets to `(B*T,)`
(or `logits.transpose(1, 2)`).

```python
loss = F.cross_entropy(logits, targets)
```
```python
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
```
Caught by: `test_loss_is_mean_token_cross_entropy`, `test_loss_backpropagates_to_all_parameters`,
`test_tied_weights_stay_identical_after_an_optimizer_step`.

### B2: no context-window crop in `generate`

The positional table has only `block_size` rows (and `forward` rejects longer inputs), so
autoregressive decoding must condition on the last `block_size` tokens only.

```python
logits, _ = self(idx)
```
```python
idx_cond = idx[:, -self.config.block_size :]
logits, _ = self(idx_cond)
```
Caught by: `test_generate_can_continue_past_the_context_window`.

### B3: residual/LayerNorm wiring

Pre-norm (GPT-2 and most modern LLMs) is `x = x + f(LN(x))`. The residual stream is never
normalized, so gradients flow through the identity path and deep stacks train stably without
warmup tricks. Post-norm (the original Transformer) is `x = LN(x + f(x))`. The PR's
`x + LN(f(x))` is neither. The attention sees raw, unnormalized activations (whose scale grows with
depth), and the PR doesn't match the design it documents (the MLP sublayer just below is correct,
which makes the bug easy to miss).

```python
x = x + self.ln_1(self.attn(x))
```
```python
x = x + self.attn(self.ln_1(x))
```
Caught by: `test_block_matches_pre_norm_reference`, `test_model_logits_match_reference_built_from_torch_primitives`.

### B4: causal mask off by one

`torch.tril(ones)` (diagonal 0) keeps `j <= i`. `diagonal=1` keeps `j <= i + 1`, so every position
can see the next token, which is exactly the label it is trained to predict. Loss collapses toward
0 in training (the PR's "loss 0.05" is a red flag), but at inference the next token does not exist
yet. Masking `diagonal=-1` would be the opposite error: row 0 would be all `-inf`, and softmax would give NaN.

```python
mask = torch.tril(torch.ones(size, size), diagonal=1)
```
```python
mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
```
Caught by: `test_future_tokens_do_not_affect_past_logits`, `test_attention_matches_scaled_dot_product_attention_reference` (+ block/model/generate reference tests).

### B5: wrong attention scale

Each score `q·k` sums `head_dim` products. With unit-variance q/k its variance is `head_dim`,
so dividing by `sqrt(head_dim)` restores unit variance. Dividing by `sqrt(n_embd)` (here
`sqrt(32)` vs `sqrt(8)`, so 2x too small) flattens the softmax, which weakens the attention
signal and changes the model's effective temperature. Nothing crashes.

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(C)
```
```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
```
Caught by: `test_attention_matches_scaled_dot_product_attention_reference`, block/model reference tests.

### B6: heads merged in the wrong memory order

`att @ v` is `(B, H, T, D)`. `reshape(B, T, C)` on it does not raise, because the element count
matches. But row `t` of the result is now a slice of head `h`'s sequence and not the concatenation of
all heads at position `t`. That mixes positions, so it also breaks causality. You must
`transpose(1, 2)` to `(B, T, H, D)` first, then `reshape` (or `.contiguous().view`; plain `.view`
after a transpose raises "view size is not compatible...").

```python
y = (att @ v).reshape(B, T, C)
```
```python
y = (att @ v).transpose(1, 2).reshape(B, T, C)
```
Caught by: `test_attention_matches_scaled_dot_product_attention_reference`, `test_future_tokens_do_not_affect_past_logits`.

### B7: weight tying broken by `clone()`

Tying means **one** `nn.Parameter` object in two places, so it gets a single gradient
(the sum of both uses) and a single optimizer state. `nn.Parameter(w.clone())` copies the tensor
and creates a new leaf. `parameters()` then yields both (V*C extra params), and the two diverge
after the first step. Here `self.apply(_init_weights)` even re-initializes the copy, so the two
differ from the start. Assign the parameter itself: `self.lm_head.weight = self.tok_emb.weight`.

```python
self.lm_head.weight = nn.Parameter(self.tok_emb.weight.clone())
```
```python
self.lm_head.weight = self.tok_emb.weight
```
Caught by: `test_lm_head_shares_the_token_embedding_parameter`, `test_tied_weights_stay_identical_after_an_optimizer_step`, `test_parameter_count_counts_the_tied_matrix_once`.

### B8: decoding with autograd on and dropout active

Called right after a training step (model in `train()` mode), `generate` samples through
dropout, so `greedy=True` gives different, worse tokens on every call. Without `no_grad`,
every step also builds an autograd graph (activations kept alive until the tensors die),
which wastes memory and time. Decorate with `@torch.no_grad()` (or `torch.inference_mode()`),
switch to `eval()`, and restore the caller's mode in a `finally`.

```python
def generate(self, idx, ...):
    for _ in range(max_new_tokens):
        logits, _ = self(idx)
```
```python
@torch.no_grad()
def generate(self, idx, ...):
    was_training = self.training
    self.eval()
    try:
        ...
    finally:
        self.train(was_training)
```
Caught by: `test_greedy_generation_ignores_dropout_and_keeps_training_mode`, `test_generate_does_not_record_autograd_history`.

## Production improvements

1. **Config validation** (applied). `GPTConfig.__post_init__` rejects `n_embd % n_head != 0` and
   dropout outside `[0, 1)`, and the dataclass is `frozen=True` so a shared config cannot be mutated.
   ```python
   if self.n_embd % self.n_head != 0:
       raise ValueError(...)
   ```
2. **Boolean, non-persistent mask buffer** (applied). `dtype=torch.bool` (no float compare per call) and
   `persistent=False`, so the derived mask stays out of checkpoints and a `block_size` change
   doesn't break `load_state_dict`.
3. **Input validation in `generate`** (applied). Reject `temperature <= 0` with a hint to use `greedy=True`.
4. **`num_parameters()`** (applied). It counts unique parameters, the standard sanity check that tying works.
5. **Use the fused kernel** (not applied, kept manual for teaching). On torch >= 2.0, `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p if self.training else 0.0)` gives Flash/memory-efficient attention, needs no mask buffer, and removes B4/B5 as bug classes.
6. **KV cache** (not applied). Decoding is O(T²) per step because it recomputes all keys and values.
   Cache the per-layer `k, v` and feed only the newest token:
   ```python
   def forward(self, x, past_kv=None):
       ...
       if past_kv is not None:
           k = torch.cat([past_kv[0], k], dim=2)
           v = torch.cat([past_kv[1], v], dim=2)
       return y, (k, v)
   ```
7. **GPT-2 residual init** (not applied). Scale `proj` / `mlp[2]` init std by `1/sqrt(2 * n_layer)`
   so the residual stream's variance does not grow with depth.
8. **Tests as specs**. Equivalence against `F.scaled_dot_product_attention`, a causality probe and
   a tying identity check (as in `test_pr.py`) belong in CI for any attention code.

## Interview talking points

- **Causality is testable.** Perturb tokens at positions `>= k` and assert logits at `< k` are
  bit-identical. Checking the loss value won't do it: a leaking mask makes loss look *better*.
- **Pre- vs post-norm.** Pre-norm keeps a clean identity residual path (`x + f(LN(x))`) and
  needs a final LayerNorm before the head. Post-norm normalizes after the add and usually needs
  LR warmup. `x + LN(f(x))` is neither.
- **Scale by `sqrt(d_k)`, not `sqrt(d_model)`.** The variance of a dot product grows with the number of
  summed terms, which is the per-head dimension.
- **Shape-legal is not the same as correct.** `reshape` without the `transpose` and `cross_entropy` on 3-D
  logits both show that matching element counts or silent broadcasting hides bugs. Compare against a
  reference built from trusted primitives, in float64.
- **Tying means identity, not equality.** Check with `a is b`. A clone has equal values on the
  first step only, and doubles the parameter count.
