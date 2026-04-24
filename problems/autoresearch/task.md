# autoresearch: GPT pretraining on a single 24 GB GPU

You are editing a single-file, single-GPU GPT pretraining program
(adapted from karpathy/autoresearch). The script trains for a fixed
wall-clock budget of `TIME_BUDGET` seconds on climbmix tokens, then
evaluates the model and prints `val_bpb` (bits per byte; lower is
better).

## What you can change

Everything in this file: model architecture, optimizer,
hyperparameters, batch size, model depth/width, training loop,
attention backend, learning-rate schedule, weight initialization.

## What is fixed (do NOT try to modify)

- Import `MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `make_dataloader`,
  and `evaluate_bpb` from `prepare.py`. That harness is the evaluation
  ground truth — don't re-implement the metric or the dataloader, and
  don't try to shadow/monkey-patch them.
- The final line must be `val_bpb: <float>` produced by calling
  `evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)`.

## Hard constraints

- **VRAM ≤ 24 GB** (RTX A5000). OOM counts as a failure (reward 0).
  The seed peaks around 6–10 GB, leaving headroom for bigger models.
- Finish eval within the Hydra `timeout` (default 300 s). Training
  itself is capped by `TIME_BUDGET` but startup + final eval also
  count.
- One NVIDIA GPU. No distributed training, no external packages beyond
  what's already in the venv.

## Reward

`reward = max(0, 2.0 - val_bpb)`. A random 8192-vocab baseline gives
val_bpb ≈ 2.6 (reward 0); a well-trained small GPT reaches val_bpb
≈ 1.0 (reward ≈ 1.0). Higher reward = lower val_bpb = better model.
Crashes, NaN losses, OOMs, and malformed output all score 0.

## Starting baseline

The seed uses:
- 6-layer GPT, n_embd=384, 3 heads, head_dim=128, full causal
  attention via `torch.nn.functional.scaled_dot_product_attention`.
- MuonAdamW optimizer (Muon for 2D matrices, AdamW otherwise).
- BPE tokenizer (vocab 8192), seq 1024, device batch 16,
  grad-accum to reach 65 536 tokens / step.
- `torch.compile` **disabled** (cold compile dominates short budgets).
  Set `COMPILE = True` if `TIME_BUDGET` is long enough for warmup to
  amortize.
- `WINDOW_PATTERN = "L"` (all full attention). Non-`L` patterns
  (e.g. `"SSSL"` banded) require you to implement custom masking
  yourself — SDPA only supports `is_causal=True`.

## Things worth exploring (not required)

- Learning rates: `MATRIX_LR`, `EMBEDDING_LR`, `UNEMBEDDING_LR`.
- LR schedule: `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`.
- Model capacity: `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`.
- Attention: GQA (`n_kv_head < n_head`), sliding windows.
- Optimizer variants: Muon `ns_steps`, momentum schedule, weight decay.
- Activations, normalization placement, value embeddings, softcap.
- Grad accumulation vs. larger device batch, precision tricks.
- Enable `torch.compile` if the time budget makes it worthwhile.

The simplicity rule: a small gain at the cost of a lot of ugly code
is usually not worth it. Deletions that preserve or improve val_bpb
are wins.
