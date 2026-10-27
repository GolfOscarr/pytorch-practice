# r05_ddp: answer key

PR: "Make our trainer multi-GPU with DistributedDataParallel"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:53 | `select_backend` returns `"nccl"` for every non-CPU device, MPS included | crash (this Mac's default device is MPS, so `main` dies in `init_process_group`) |
| B2 | High | pr.py:98-101 | `sampler.set_epoch(epoch)` is never called; `epoch` is accepted but ignored | silently wrong (identical shuffle every epoch) |
| B3 | Critical | pr.py:99 | Sampler seed is `seed + rank`, so each rank uses a different permutation | silently wrong (shards overlap, part of the data is never trained on) |
| B4 | High | pr.py:153 | `reduce_metrics` assumes every rank saw `count` samples (`count * world_size`) | silently wrong (wrong global mean; ranks report different numbers) |
| B5 | High | pr.py:108-111 | Validation uses `DistributedSampler`, which pads by repeating samples | silently wrong (some val samples counted twice) |
| B6 | Medium | pr.py:161 (called at :207) | Checkpoint stores `ddp_model.state_dict()`, keys all prefixed `module.` | crash (`load_checkpoint` into a plain model fails) |
| B7 | High | pr.py:206-207 | Every rank writes the same checkpoint file, with no barrier | concurrency (races and a possibly corrupt file; N times the I/O) |
| B8 | Medium | pr.py:188-209 | `cleanup_process_group()` runs only on the happy path, with no `try/finally` | resource leak (process group, sockets and store left open after an exception) |

### B1: NCCL picked for a device it doesn't support

NCCL only works with CUDA. MPS, CPU and XPU must use gloo (or a vendor backend). The
condition `device.type == "cpu"` treats every accelerator as CUDA. On this Mac,
`resolve_device` returns `mps` by default, so the PR crashes on the laptops its description
says it supports. Choose the backend by checking for the one thing that requires it (CUDA),
not by excluding the one thing that doesn't.

Picking gloo is necessary but not sufficient on a Mac: **no c10d backend supports MPS tensors**
(gloo's `all_reduce` on an MPS tensor raises `NotImplementedError`, and so does DDP's
construction-time broadcast), and MPS has no float64 for the metric reduction either. So a
non-CUDA distributed run must train on CPU; fixed.py's `resolve_device` does that.

```python
return "gloo" if device.type == "cpu" else "nccl"  # buggy
return "nccl" if device.type == "cuda" else "gloo"  # fixed
```

Caught by `test_backend_matches_device[mps-gloo]`.

### B2: `set_epoch` never called

`DistributedSampler` builds its permutation from `seed + self.epoch`. `self.epoch` stays 0
unless you call `set_epoch`, so every epoch iterates the same order and the same batches.
SGD loses the decorrelation it relies on. Nothing crashes, and the loss still goes down,
just a bit worse. The function signature even takes `epoch`, which never gets used.

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, ...)
return DataLoader(dataset, batch_size=batch_size, sampler=sampler)          # buggy

sampler.set_epoch(epoch)                                                   # fixed
return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```

Caught by `test_train_order_changes_between_epochs`.

### B3: per-rank sampler seed

Every rank shuffles the full index list with the same generator, then takes the strided
slice `indices[rank::world_size]`. Only a shared permutation makes the slices a partition.
With `seed + rank`, each rank slices a different permutation. The shards overlap, some
samples get seen twice per epoch, and others are never seen. With 2 ranks, each one takes an
effectively random half, so about a quarter of the data is never seen in a given epoch. No
metric or error shows it. If you want per-rank randomness
(augmentation, dropout), seed those RNGs per rank and keep the sampler seed shared.

```python
DistributedSampler(dataset, ..., shuffle=True, seed=seed + rank)  # buggy
DistributedSampler(dataset, ..., shuffle=True, seed=seed)  # fixed
```

Caught by `test_train_shards_partition_the_dataset[0|1|2]`.

### B4: global mean assumes equal per-rank counts

The global mean is `sum(all sums) / sum(all counts)`. `count * world_size` matches that only
when every rank processed exactly `count` samples. That breaks with uneven eval shards, a
partial last batch under `drop_last=False` without padding, or `Join`-style uneven inputs.
Each rank then divides the same global sum by a different denominator, so ranks log
different "global" numbers. Fix: pack the count into the same tensor and reduce sums and
count in one collective.

```python
totals = torch.tensor([sums[k] for k in keys], ...)
dist.all_reduce(totals)
global_count = count * dist.get_world_size()                                     # buggy

packed = torch.tensor([*(sums[k] for k in keys), count], dtype=torch.float64, ...)
dist.all_reduce(packed)
global_count = packed[-1].item()                                                 # fixed
```

Caught by `test_reduce_metrics_weights_each_rank_by_its_sample_count`. With B5 fixed, also
caught by `test_every_rank_reports_the_same_metrics` and the full-val-set test.

### B5: padded validation sharding double-counts samples

When `drop_last=False`, `DistributedSampler` pads the index list up to a multiple of
`world_size` by repeating leading indices. With 9 val samples and 2 ranks, sample 0 gets
evaluated twice. Training needs that padding, because every rank must run the same number
of steps or DDP's all-reduce deadlocks. A metric must not have it. `drop_last=True` is no
fix either, since it drops samples. Shard eval without padding, reduce sums and counts
(B4), and run eval on the unwrapped module so uneven step counts can't hang a DDP
collective.

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
return DataLoader(dataset, batch_size=batch_size, sampler=sampler)  # buggy

shard = Subset(dataset, range(rank, len(dataset), world_size))
return DataLoader(shard, batch_size=batch_size, shuffle=False)  # fixed
```

Caught by `test_eval_shards_cover_every_sample_exactly_once` and
`test_reported_val_loss_equals_single_process_loss_on_full_val_set`.

### B6: `module.` prefix in the checkpoint

DDP registers the wrapped model as the submodule `module`, so `ddp_model.state_dict()` has
keys like `module.0.weight`. `load_checkpoint` documents loading into a plain model, which
is the normal case for inference, eval jobs, and resuming before re-wrapping. That load
fails with "Missing key(s) ... Unexpected key(s)". Save `ddp_model.module.state_dict()`.
Being tolerant on load (unwrap a DDP target) also helps.

```python
"model": model.state_dict()                 # buggy (model is the DDP wrapper)
"model": _unwrap(model).state_dict()        # fixed
```

Caught by `test_checkpoint_loads_into_a_plain_model`, and transitively by the full-val-set test.

### B7: every rank writes the checkpoint

DDP keeps replicas identical, so one writer is enough. With N writers, `torch.save` calls
interleave on the same path. On shared or network storage that can corrupt the file, it
costs N times the I/O, and any reader racing the writers sees a partial file. The fix is
`if rank == 0: save(...)` followed by `dist.barrier()`, so no rank moves on (for example to
load or evaluate the file) before it is complete. With the atomic `tmp` + `os.replace` write
in fixed.py, concurrent writers can even crash: with some interleavings the second
`os.replace` finds its tmp file already moved.

```python
if cfg.checkpoint_path is not None:
    save_checkpoint(cfg.checkpoint_path, ddp_model, optimizer, epoch)  # buggy

if cfg.checkpoint_path is not None:
    if rank == 0:
        save_checkpoint(cfg.checkpoint_path, ddp_model, optimizer, epoch)
    dist.barrier()  # fixed
```

Caught by `test_only_rank_zero_writes_checkpoints`.

### B8: process group not torn down on error

`cleanup_process_group()` is the last statement of `main`, so any exception (a bad sample,
OOM, a KeyboardInterrupt) skips it. The process group, its TCP or file store, and the gloo
or NCCL communicators leak. In long-lived processes (notebooks, test runners, elastic agents
that restart the worker function) the next `init_process_group` fails with "trying to
initialize the default process group twice". NCCL also warns or hangs at interpreter exit.
Wrap everything after setup in `try/finally`.

```python
setup_process_group(...)
...training...
cleanup_process_group()                       # buggy: skipped on any exception
return history

setup_process_group(...)
try:
    ...training...
    return history
finally:
    cleanup_process_group()                   # fixed
```

Caught by `test_training_error_propagates_and_process_group_is_destroyed`.

## Production improvements

1. **Atomic checkpoint writes** *(applied in fixed.py)*. Write to `path.tmp`, then
   `os.replace`. A crash mid-save never leaves a truncated "latest" file.
   ```python
   torch.save(state, f"{path}.tmp")
   os.replace(f"{path}.tmp", path)
   ```
2. **No per-step host sync** *(applied)*. `loss.item()` inside the loop blocks on the GPU
   every step. Accumulate on device and call `.item()` once per epoch.
   ```python
   loss_sum += loss.detach() * x.size(0)  # ... return loss_sum.item(), seen
   ```
3. **Evaluate the unwrapped module** *(applied)*. `evaluate(model, ...)`, not
   `evaluate(ddp_model, ...)`. Eval shards are uneven, and a DDP forward can issue buffer
   broadcasts that deadlock when ranks run different numbers of steps.
4. **Rank-0-only logging plus config validation** *(applied)*. `log.info` only on rank 0.
   `@dataclass(frozen=True)` with `__post_init__` rejects bad `epochs`, `batch_size` or `lr`
   before any process group exists.
5. **Collective timeouts and a debug mode** *(not applied)*. Pass
   `timeout=timedelta(minutes=5)` to `init_process_group`, and set
   `TORCH_DISTRIBUTED_DEBUG=DETAIL` in CI. A mismatched collective then fails loudly
   instead of hanging the job until the scheduler kills it.
6. **Resume support that keeps ranks in sync** *(not applied)*. Load on every rank before
   DDP wrapping (or on rank 0 plus `broadcast_object_list`), and restore `start_epoch`, the
   optimizer state, and `sampler.set_epoch(start_epoch)`, so resumed runs replay the
   un-resumed shuffle sequence.
   ```python
   start = load_checkpoint(path, model, optimizer) if resume else 0
   ddp_model = DDP(model)  # DDP broadcasts rank-0 params at construction
   ```
7. **Launch through `torchrun` and env vars** *(not applied)*. Read `RANK`, `LOCAL_RANK`
   and `WORLD_SIZE` from the environment, so the same `main` runs under `torchrun`, SLURM
   or k8s. Map devices by `LOCAL_RANK`, not global rank, on multi-node jobs.
8. **DataLoader throughput** *(not applied)*. `num_workers>0`, `pin_memory=True` and
   `persistent_workers=True` on CUDA, with `non_blocking=True` copies (already added in
   fixed.py).

## Interview talking points

- **Sampler contract.** `DistributedSampler` is a *shared* permutation sliced by rank. The
  seed must be the same on all ranks, and `set_epoch` must be called every epoch. Different
  seeds break the partition (B3). A missing `set_epoch` freezes the order (B2). Neither
  raises an error, so you check it with a test that the shards partition the dataset.
- **Padding is for training, not for metrics.** Training ranks must take equal step counts,
  or DDP's gradient all-reduce hangs, and that's why the sampler pads. Metrics must count
  every sample exactly once. So shard eval without padding and all-reduce `(sum, count)`
  together, never a mean of means or `count * world_size` (B4, B5).
- **Single-writer side effects.** Only rank 0 writes checkpoints, logs and metrics files.
  Follow the write with a barrier if other ranks depend on it. Save `model.module`'s
  state_dict so the artifact doesn't depend on how it was trained (B6, B7).
- **Backend and lifecycle.** NCCL is CUDA-only; gloo covers CPU (and CUDA) and CI. MPS has no
  distributed backend at all, so multi-process runs on a Mac train on CPU. Put
  `init_process_group` and `destroy_process_group` in a `try/finally`, the way you'd treat
  any OS resource (B1, B8).
- **How I'd test DDP without GPUs.** Spawn 2 CPU processes over gloo with a `file://`
  rendezvous in a temp dir (no port collisions), write per-rank results to files, and
  assert in the parent. Add a join timeout so a deadlock fails the test instead of hanging
  CI.
