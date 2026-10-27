# r09_lora — answer key

PR: "Add LoRA adapters for parameter-efficient fine-tuning"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:112 | `name.endswith(targets)` matches `kv_proj` when targeting `v_proj` | silently wrong (extra modules adapted, budget/checkpoint wrong) |
| B2 | High | pr.py:45 | `scaling = r / alpha` instead of `alpha / r` | silently wrong (adapter 4x too weak at r=8, alpha=16; LR coupling to r inverted) |
| B3 | Critical | pr.py:60 | `lora_B` initialized with `normal_(std=0.02)` instead of zeros | silently wrong (model differs from the pretrained one at step 0) |
| B4 | Critical | pr.py:64 | `delta_weight` is `A.T @ B.T` = `(B @ A).T`, the transpose of the real update | crash on non-square layers; silently wrong merged model on square layers |
| B5 | Medium | pr.py:76 | `merge()` has no `if self.merged: return` guard | silently wrong (second merge adds the delta twice; `unmerge` only removes one) |
| B6 | High | pr.py:67 | adapter dropout applied to the input of the frozen base path too | silently wrong (training forward perturbs pretrained features; train/eval mismatch) |
| B7 | Critical | pr.py:123-125 | only `base.weight` of wrapped layers frozen; biases, embeddings, norms, `k_proj`/`o_proj`/MLP/head stay trainable | silently wrong + perf (full fine-tune of everything else, optimizer state for the whole model) |
| B8 | Medium | pr.py:149 | `lora_state_dict` returns tensors aliasing the live parameters | silently wrong (snapshots/"best checkpoint" mutate as training continues) |

### B1: Target matching by suffix hits `kv_proj`

`str.endswith` with a tuple is idiomatic-looking, but it is a suffix match on the whole dotted path,
so `"blocks.0.cross_attn.kv_proj".endswith("v_proj")` is `True`. Any fused projection (`kv_proj`,
`qkv_proj`) or name like `dev_proj` gets wrapped too. Match on the leaf component exactly (that is what
PEFT does for plain strings).

```python
if isinstance(module, nn.Linear) and name.endswith(targets)
```
```python
if isinstance(module, nn.Linear) and name.rpartition(".")[2] in targets
```
Caught by: `test_inject_wraps_exactly_the_target_modules`, `test_only_lora_params_require_grad`,
`test_trainable_param_count_is_r_times_in_plus_out_per_target`, `test_lora_state_dict_contains_only_adapter_tensors`.

### B2: Scaling is `r / alpha`

LoRA uses `h = W x + (alpha / r) B A x`. Dividing by `r` keeps the update magnitude roughly constant
when you change rank, so you don't retune LR each time; `alpha` is the knob. `r / alpha` makes the
update scale *grow* with rank and shrink with alpha — the opposite of every config you'll copy from a
paper or PEFT.

```python
self.scaling = r / alpha
```
```python
self.scaling = alpha / r
```
Caught by: `test_forward_matches_low_rank_formula_with_alpha_over_r_scaling`, `test_dropout_is_disabled_in_eval_mode`.

### B3: `B` initialized randomly

The key LoRA invariant: at initialization `B A = 0`, so the adapted model is *exactly* the pretrained
model and fine-tuning starts from it. Standard init is A ~ Kaiming, B = 0 (A random so B gets a non-zero
gradient on step 1; B zero so the output is unchanged). With both random the model starts from
pretrained + noise — loss on step 0 is already worse. The perturbation is small (std 0.02), but the invariant that
makes LoRA safe (adapted model == base model at step 0) is broken.
(If both were zero, nothing would ever learn: dL/dA ∝ B and dL/dB ∝ A x.)

```python
nn.init.normal_(self.lora_B, std=0.02)
```
```python
nn.init.zeros_(self.lora_B)
```
Caught by: `test_fresh_adapter_does_not_change_layer_output`, `test_injected_model_matches_base_model_at_init`,
`test_adapter_dropout_does_not_touch_base_path_in_training`.

### B4: Merge uses the transposed delta

`nn.Linear` stores `weight` as `(out, in)` and computes `x @ W.T`. Forward computes `x @ A.T @ B.T`
= `x @ (B A).T`, so the dense update to add to `W` is `B @ A` (shape `(out, in)`). The PR wrote
`A.T @ B.T`, mirroring the forward expression, which is `(B A).T` — shape `(in, out)`. On a non-square
layer `+=` raises a broadcast error; on a square layer (`q_proj`, `v_proj` with `d -> d` — exactly the
default targets!) it silently merges the wrong matrix, so the exported model differs from the trained one.

```python
return (self.lora_A.T @ self.lora_B.T) * self.scaling
```
```python
return (self.lora_B @ self.lora_A) * self.scaling
```
Caught by: `test_merged_output_matches_unmerged_output` (crash), `test_merged_output_matches_for_square_layer`,
`test_model_merge_matches_unmerged_output`, `test_merging_twice_does_not_change_output`.

### B5: `merge()` is not idempotent

`unmerge()` checks the flag but `merge()` doesn't. Merging is often triggered from several places
(an export script, a `merge_and_unload` helper, an eval hook); a second call adds `ΔW` again while
`merged` was already `True`, and a later `unmerge()` removes only one copy. In-place weight surgery
needs a state flag *checked on both sides*.

```python
def merge(self) -> None:
    self.base.weight += self.delta_weight()
    self.merged = True
```
```python
def merge(self) -> None:
    if self.merged:
        return
    self.base.weight += self.delta_weight()
    self.merged = True
```
Caught by: `test_merging_twice_does_not_change_output`.

### B6: Dropout on the base path

LoRA dropout regularizes only the adapter branch: `W x + s · B A dropout(x)`. Reassigning `x =
self.dropout(x)` first means the frozen pretrained layer sees dropped-and-rescaled inputs in training
mode too: training no longer starts from the pretrained function (even with B=0), and the gradients the
adapter receives are computed against a perturbed base. In eval mode dropout is identity, so the bug
only shows during training — classic train/eval discrepancy.

```python
x = self.dropout(x)
result = self.base(x)
...
return result + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```
```python
result = self.base(x)
...
return result + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
```
Caught by: `test_adapter_dropout_does_not_touch_base_path_in_training`.

### B7: Freezing only the wrapped layers' weights

"Only LoRA trainable" means *everything* else is frozen: the wrapped layers' biases, embeddings,
LayerNorms, untargeted Linears (`k_proj`, `o_proj`, MLP, LM head). The PR freezes just
`base.weight` of wrapped modules, so this is effectively full fine-tuning plus adapters: the
memory/optimizer-state savings vanish (`build_optimizer` will happily include all those params),
base weights drift so the adapter checkpoint alone can't reproduce the model, and
`count_trainable_params` reports the wrong budget.

```python
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.base.weight.requires_grad_(False)
```
```python
for name, param in model.named_parameters():
    param.requires_grad_("lora_" in name)
```
Caught by: `test_only_lora_params_require_grad`, `test_gradients_flow_only_to_adapters`,
`test_trainable_param_count_is_r_times_in_plus_out_per_target`, `test_optimizer_param_groups_hold_only_trainable_params_with_decay_policy`,
`test_training_step_changes_only_adapters`, `test_adapter_snapshot_can_roll_back_training`.

### B8: Adapter snapshot aliases live parameters

`Module.state_dict()` returns `param.detach()` — a new tensor object that **shares storage** with the
parameter. `optimizer.step()` updates parameters in place, so a dict taken at step 100 ("best so far",
"rollback point", EMA seed) silently shows step-N values later. `torch.save` immediately is fine, which
is why this survives casual testing. Clone when the snapshot is meant to outlive the next step.

```python
return {k: v for k, v in model.state_dict().items() if "lora_" in k}
```
```python
return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
```
Caught by: `test_adapter_snapshot_can_roll_back_training`.

## Production improvements

1. **Strict adapter loading** (applied in fixed.py). `load_state_dict(state, strict=False)` with the
   result discarded silently loads *nothing* if the adapter came from a different target set or model.
   Validate keys against the model's own `lora_` keys:
   ```python
   expected = {k for k in model.state_dict() if "lora_" in k}
   missing, unexpected = expected - state.keys(), state.keys() - expected
   if missing or unexpected:
       raise ValueError(f"adapter keys mismatch: missing={...}, unexpected={...}")
   ```
2. **Fail loudly on zero matches** (applied). A typo in `target_modules` should raise, not train a
   frozen model: `if not to_replace: raise ValueError(...)`.
3. **Config validation + logging** (applied). Validate `0 <= dropout < 1`; `logger.info` the number of
   wrapped modules so the parameter budget shows up in run logs.
4. **Dataclass config** (not applied). Put `r`, `alpha`, `dropout`, `target_modules` in a frozen
   `@dataclass LoRAConfig` and save it next to the adapter so loading can reconstruct the injection:
   ```python
   @dataclass(frozen=True)
   class LoRAConfig:
       r: int = 8
       alpha: float = 16.0
       dropout: float = 0.0
       target_modules: tuple[str, ...] = ("q_proj", "v_proj")
   ```
5. **Exact unmerge** (not applied). `W + ΔW - ΔW` is not bit-exact in bf16/fp16. Either stash the
   original weight on merge (costs memory) or merge only into a copy for export
   (`merged = copy.deepcopy(model); merge_all(merged)`), never into the training model.
6. **`merge_and_unload(model)` helper** (not applied). Merge every `LoRALinear` and replace it with its
   `base` `nn.Linear`, so the exported model has no LoRA classes and no runtime overhead — what an
   inference server actually wants.
7. **Guard training while merged** (not applied). Gradient through a merged layer goes nowhere useful;
   raise in `forward` if `self.training and self.merged`, or auto-unmerge on `.train()`.
8. **Save with `safetensors`/`weights_only=True`** (not applied). Adapters get shared around; loading
   them with `torch.load(weights_only=True)` (or safetensors) avoids arbitrary-code pickle execution.

## Interview talking points

- **Init invariant:** "LoRA must start as the identity edit: B = 0, A random. B = 0 makes the output
  equal the pretrained model; A random makes dL/dB non-zero so learning starts. I'd assert
  `adapted(x) == base(x)` at init in a unit test."
- **Shapes of the merge:** "nn.Linear stores (out, in) and computes x Wᵀ; the forward is x Aᵀ Bᵀ =
  x (BA)ᵀ, so ΔW = s·B A. Copying the forward expression gives the transpose — and on square q/v
  projections that doesn't even crash. Always test merged == unmerged numerically."
- **Scaling = alpha / r:** "It decouples update magnitude from rank so you can sweep r without retuning
  the LR; alpha is the effective strength."
- **What 'freeze' really means:** "requires_grad on every non-adapter parameter, then build the
  optimizer from `requires_grad` params only, then check `count_trainable_params == Σ r(in+out)`.
  If the count is off, either freezing or module matching is wrong — both bugs here show up in that one number."
- **Aliasing:** "state_dict tensors share storage with parameters; in-place optimizer steps mutate them.
  Clone for in-memory snapshots, and match module names on the leaf, not by suffix."
