# r11_contrastive: answer key

PR: "Add CLIP-style contrastive (InfoNCE) training objective"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:46 | `LayerNorm(embedding_dim)` applied to a `projection_dim`-wide tensor | crash (any head where in-dim != out-dim) |
| B2 | Critical | pr.py:114 | Image embeddings L2-normalized along `dim=0` (the batch), not per sample | silently wrong |
| B3 | High | pr.py:130 | t2i loss reuses `logits_per_image`, so only image->text is trained | silently wrong |
| B4 | Medium | pr.py:119 | Clamp applied to the *log* scale (`clamp(max=100).exp()`), so the real cap is e^100 | silently wrong (instability) |
| B5 | Critical | pr.py:124 | Labels are `arange(local_batch)`, missing the `rank * local_batch` offset | silently wrong (multi-GPU) |
| B6 | Critical | pr.py:71-73 | Gather runs fully under `no_grad`; the local shard is never put back, so no gradient flows through the gathered side | silently wrong (multi-GPU) |
| B7 | Medium | pr.py:84 | i2t accuracy does `argmax(dim=0)` (over queries) instead of over candidates | silently wrong metric; crash when world size > 1 |
| B8 | Medium | pr.py:52 | `F.dropout(x, p)` defaults to `training=True`, so dropout still runs after `.eval()` | silently wrong / non-reproducible inference |

### B1: LayerNorm sized for the input width

`LayerNorm(d)` normalizes over the last dimension and its affine weights have shape `[d]`. The
residual `x + projected` has width `projection_dim`, so the norm has to be sized to that.
It only "works" when `embedding_dim == projection_dim`, which is why a smoke run with matching
dims would not have caught it.

```python
self.layer_norm = nn.LayerNorm(embedding_dim)  # buggy
self.layer_norm = nn.LayerNorm(projection_dim)  # fixed
```

Caught by: `test_projection_head_maps_to_projection_dim`.

### B2: normalizing across the batch

CLIP compares directions: each embedding is put on the unit sphere and the logit is
`scale * cos(img_i, txt_j)`. `F.normalize(x, dim=0)` instead divides every *feature column* by its
norm over the batch. Rows are no longer unit vectors, so a logit depends on the other samples in
the batch and on each sample's magnitude, and the temperature stops meaning anything. The text
side used `dim=-1`, so the two towers are not even normalized the same way.

```python
image = F.normalize(image_emb, dim=0)  # buggy
image = F.normalize(image_emb, dim=-1)  # fixed
```

Caught by: `test_loss_matches_reference_infonce`, `test_loss_is_invariant_to_per_sample_embedding_scale`,
`test_initial_logit_scale_is_inverse_temperature`, `test_loss_is_symmetric_in_image_and_text`,
`test_logit_scale_is_capped_at_max`, `test_gradient_matches_reference_...`.

### B3: only one direction of the symmetric loss

CLIP averages two cross-entropies: image->text (rows of `img @ txt.T`) and text->image (rows of
`txt @ img.T`, which is the transpose when world size is 1). A copy-paste left
`logits_per_image` in both terms, so the loss is just image->text counted twice. It still goes
down, so a smoke run looks fine, but text-as-query retrieval gets no training signal and the
loss is no longer symmetric under swapping modalities.

```python
loss_t2i = F.cross_entropy(logits_per_image, labels)  # buggy
loss_t2i = F.cross_entropy(logits_per_text, labels)  # fixed
```

Caught by: `test_loss_matches_reference_infonce`, `test_loss_is_symmetric_in_image_and_text`,
`test_gradient_matches_reference_when_gather_returns_detached_copies`.

### B4: clamp applied to the log-parameter

The parameter is `log(scale)`. The paper caps the *scale* at 100 (temperature >= 0.01).
`logit_scale.clamp(max=100).exp()` caps the log at 100, so the scale can reach e^100. The
learnable temperature is known to drift upward, and when it does, logits blow up, the softmax
saturates and the loss goes NaN/inf (fp16/bf16 hits this first). Either clamp after `exp`, or
clamp at `log(max_scale)`.

```python
scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()  # buggy
scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)  # fixed
```

(`torch.clamp` in the graph is fine: it passes gradient through below the cap and zero above it.
An in-place `self.logit_scale.clamp_()` outside `no_grad` would raise instead, and OpenCLIP's
alternative clamps the parameter under `no_grad` after `optimizer.step()`.)

Caught by: `test_logit_scale_is_capped_at_max`.

### B5: labels not offset by rank

With gathering, rank `r`'s logits have shape `[local_batch, world * local_batch]` and its
positives sit in columns `r*B .. r*B + B-1`. `arange(B)` is only right on rank 0. Every other rank
trains its images to match rank 0's captions. Single-GPU runs look perfect, and multi-GPU runs
quietly learn noise on (world-1)/world of the data.

```python
labels = torch.arange(local_batch, device=image.device)  # buggy
labels = torch.arange(local_batch, device=image.device) + self.rank * local_batch  # fixed
```

Caught by: `test_labels_point_at_local_rows_of_global_batch[1]`,
`test_aligned_pairs_on_any_rank_have_low_loss_and_perfect_retrieval[1]`.

### B6: gathered features carry no gradient for the local shard

`dist.all_gather` writes into fresh buffers and is not differentiable, so the gathered tensors are
constants. The minimal fix is to put the live local tensor back at `shards[rank]`. Then
`logits_per_text = text @ all_image.T` sends gradient to this rank's images again, for every
term of **this rank's** loss.

That is still not the exact full-batch gradient. This rank's samples are also negatives in every
other rank's loss, and those ranks hold our features as constants, so those gradient terms are
dropped; after DDP averages gradients you get an approximation. The two exact options:
- **Global logits on every rank**: every rank computes the full `[W·B, W·B]` logit matrix
  (OpenCLIP's `gather_with_grad=False` together with `local_loss=False`). The live local shard
  then receives every gradient term, at the cost of W× more logit compute.
- **A differentiable all-gather** (`torch.distributed.nn.functional.all_gather`, OpenCLIP's
  `gather_with_grad=True`): its backward reduce-scatters gradients back to the owning rank, so
  local logits stay exact.

The PR wrapped the gather in `no_grad` and never re-inserted the shard, so even this rank's own
loss sends no gradient to the gathered side. It only
works on one GPU because `default_gather` returns the input object itself there. The `rank`
argument goes unused, which is a hint.

```python
with torch.no_grad():  # buggy
    shards = list(gather_fn(features))
return torch.cat(shards, dim=0)

shards = list(gather_fn(features))  # fixed
shards[rank] = features
return torch.cat(shards, dim=0)
```

Caught by: `test_gradient_matches_reference_when_gather_returns_detached_copies`.

### B7: retrieval accuracy argmax over the wrong dimension

Rows are queries and columns are candidates. "Which caption does image i retrieve" is
`argmax(dim=-1)`. `argmax(dim=0)` answers "which image is most similar to caption j", which is
the t2i question, and on a square matrix it silently reports a different metric. On a rectangular
`[B, world*B]` matrix the shapes don't broadcast against `labels`, so it crashes in multi-GPU eval.

```python
i2t = (logits_per_image.argmax(dim=0) == labels).float().mean()  # buggy
i2t = (logits_per_image.argmax(dim=-1) == labels).float().mean()  # fixed
```

Caught by: `test_retrieval_accuracy_scores_each_query_row`,
`test_aligned_pairs_on_any_rank_have_low_loss_and_perfect_retrieval`.

### B8: functional dropout ignores eval mode

`F.dropout(input, p, training=True)` defaults `training` to True, and it does not read the
module's `self.training`. After `model.eval()` the head still drops activations, so embeddings
for retrieval indexing are random, and val metrics are noisy and not reproducible. Use
`nn.Dropout` (it respects `.train()/.eval()`) or pass `training=self.training`.

```python
x = F.dropout(x, p=self.dropout)  # buggy
self.dropout = nn.Dropout(dropout)  # fixed (in __init__)
x = self.dropout(x)  # fixed (in forward)
```

Caught by: `test_projection_head_is_deterministic_in_eval_mode`.

## Production improvements

1. **Input validation** *(applied in fixed.py)*: reject mismatched or non-2D embeddings with a
   clear message instead of an opaque matmul error deep in the graph.
   ```python
   if image_emb.ndim != 2 or image_emb.shape != text_emb.shape:
       raise ValueError(f"expected matching [batch, dim] embeddings, got ...")
   ```
2. **Validate the gather contract** *(applied)*: check `0 <= rank < world_size` and that every
   shard has the local shape. Uneven last batches with `drop_last=False` would otherwise mis-place
   labels or crash in `all_gather`.
3. **`@torch.no_grad()` as a decorator on metrics** *(applied)*: on both `retrieval_accuracy` and
   `retrieval_metrics`, so callers can't accidentally build a graph during logging.
4. **Config dataclass**: group `init_temperature`, `max_logit_scale`, `local_loss`,
   `gather_with_grad` into a frozen `@dataclass(frozen=True) class ClipLossConfig`, validated in
   `__post_init__`, and logged with the run config.
5. **Label cache**: OpenCLIP caches `labels` per `(device, local_batch)` to avoid an
   allocation/H2D every step:
   ```python
   key = (logits.device, n)
   if key not in self._labels:
       self._labels[key] = torch.arange(n, device=logits.device) + self.rank * n
   ```
6. **Keep logits in fp32 under autocast**: compute `scale * img.float() @ txt.float().T` (or
   wrap in `torch.autocast(enabled=False)`). Logits up to 100 times bf16 rounding error noticeably
   hurt the softmax.
7. **Observability**: return or log `logit_scale.exp().item()` (temperature) each N steps.
   A temperature pinned at the cap is the earliest sign of collapse or an LR problem.
8. **Option for full-gradient gather**: add `gather_with_grad: bool` that uses
   `torch.distributed.nn.functional.all_gather` for exact gradients across ranks, and document the
   memory/comm trade-off against the "local shard only" default.

## Interview talking points

- **Follow-up you should expect on gathering**: "Is re-inserting the local shard exact?" Only if
  every rank computes global logits; with local logits it drops the gradient of other ranks'
  losses w.r.t. your samples. Exact alternatives: global logits, or a differentiable all-gather.

- **Normalization + temperature are the whole model of similarity.** InfoNCE logits are
  `cos(img_i, txt_j) / tau`. Normalizing along the wrong axis (B2) or capping the log instead of
  the scale (B4) silently changes what is optimized. Explain why `tau` is log-parameterized
  (positivity, scale-free updates) and why it is capped (prevents saturating the softmax).
- **The loss is symmetric by design.** Two cross-entropies, rows and columns, averaged. A
  copy-paste that trains one direction (B3) still converges, so only a reference-implementation
  test or a swap-symmetry test catches it.
- **Distributed contrastive learning has two classic traps**: gathered tensors are constants, so
  re-insert the local shard to keep gradients (B6), and each rank's positives sit at
  `rank * B + arange(B)` in the global batch (B5). Both are invisible on one GPU, so test them
  with an injected fake gather instead of a real process group.
- **Train/eval mode bugs** (B8): functional ops like `F.dropout` don't see `module.training`.
  Prefer `nn.Dropout`, and test that `eval()` output is deterministic.
- **Metrics need the same axis discipline** (B7): rows are queries and columns are candidates.
  Testing on a square matrix hides axis bugs, so use asymmetric or rectangular fixtures.
