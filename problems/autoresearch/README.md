# autoresearch — single-GPU GPT pretraining

Port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
as an SMCEvolve problem. The evolved program is a single-file GPT
pretraining script (`train.py` upstream, `initial_program.py` here).
Each candidate trains for a fixed wall-clock budget on climbmix
tokens and is scored by `val_bpb` (bits per byte; lower = better).

## What changed vs. upstream

Upstream targets an H100 (80 GB). This port targets a single RTX
A5000 (24 GB).

| Change | Upstream | Here | Why |
|---|---|---|---|
| Attention backend | Flash-Attention 3 (Hopper-only) | `F.scaled_dot_product_attention` | Works on any recent CUDA GPU; drops the `kernels` package. |
| `MAX_SEQ_LEN` | 2048 | 1024 | Halves activation memory. |
| `DEVICE_BATCH_SIZE` | 128 | 16 | Fits 24 GB. |
| `TOTAL_BATCH_SIZE` | 2**19 (~524 K) | 2**16 (~65 K) | Keeps tokens/step reasonable for short budgets. |
| `DEPTH` | 8 | 6 | ~26 M params instead of ~50 M. |
| `WINDOW_PATTERN` | `"SSSL"` | `"L"` | SDPA doesn't support banded windows; full causal is the seed. |
| `TIME_BUDGET` | 300 s | 60 s (`$AR_TIME_BUDGET`) | Many SMC particles × 5 min is too expensive by default; override for serious runs. |
| `EVAL_TOKENS` | ~20 M | ~262 K | Keeps the final bpb eval under ~10 s. |
| `torch.compile` | on | off | Cold-compile (~60 s) dominates the small default budget. Flip `COMPILE = True` if you raise `AR_TIME_BUDGET`. |

Everything else (Muon + AdamW optimizer, value embeddings, softcap, LR
schedule shape, rotary embeddings, polar-express orthogonalization) is
verbatim from upstream.

## Interface alignment with SMCEvolve

Standard SMCEvolve problem layout:

```
problems/autoresearch/
├── task.md                 # LLM prompt
├── initial_program.py      # seed train.py — the evolved file
├── evaluator.py            # evaluate(program: str) -> float
├── prepare.py              # FIXED harness (tokenizer + eval + dataloader)
├── requirements.txt        # extra Python deps
├── smoke_test.py           # run the seed end-to-end once
└── README.md
```

Registered Hydra config: [`configs/problem/autoresearch.yaml`](../../configs/problem/autoresearch.yaml).

The evaluator executes the candidate in-process via `exec()` (safe
because SMCEvolve's `_eval_runner.py` already spawns a fresh
subprocess per candidate — each CUDA context is created and torn
down per evaluation). It parses the final `val_bpb: <float>` line
and returns `max(0, 2.0 - val_bpb)` (higher = better, 0.0 on any
failure / OOM / NaN / malformed output). The 2.0 offset keeps random
baselines (bpb ≈ 2.6 with vocab 8192) pinned at 0 and lets well-trained
models hit ≈ 1.0 — consistent with other SMCEvolve reward scales.

## Environment setup

From the repo root.

### 1. Base SMCEvolve venv (once per repo)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # if you don't have uv
uv sync
source .venv/bin/activate
```

### 2. Extra packages for this problem

```bash
uv pip install -r problems/autoresearch/requirements.txt
```

This adds `rustbpe`, `tiktoken`, `pyarrow`, `requests` (all pure
Python wheels or tiny Rust extensions — no CUDA build step).

> **No FA3 / `kernels` package needed.** The seed uses
> `torch.scaled_dot_product_attention` for broad GPU compatibility.
> If you want to re-introduce FlashAttention for the agent to
> benchmark, install `flash-attn` separately and import it inside
> `initial_program.py`.

### 3. One-time data + tokenizer prep

Downloads 10 parquet shards (~1 GB) from the
`karpathy/climbmix-400b-shuffle` HuggingFace dataset and trains an
8192-vocab BPE tokenizer into `~/.cache/autoresearch/`.

```bash
python problems/autoresearch/prepare.py
# → ~2–5 minutes on a decent network
```

To fetch fewer shards (testing):

```bash
python problems/autoresearch/prepare.py --num-shards 2
```

Cache layout (reused between runs):

```
~/.cache/autoresearch/
├── data/shard_00000.parquet … shard_06542.parquet
└── tokenizer/
    ├── tokenizer.pkl
    └── token_bytes.pt
```

## Run

### Smoke test (one seed evaluation)

```bash
python problems/autoresearch/smoke_test.py --timeout 300
```

Expected output on an A5000 with the default 60 s training budget:

```
reward = 0.53     (elapsed ~74 s)     # val_bpb ≈ 1.46; reward = 2.0 - val_bpb
                                      # 158 steps, peak VRAM ≈ 4.7 GB
```

A reward of 0.0 means the seed failed — check
`~/.cache/autoresearch/` exists and `uv pip list | grep torch` shows
torch ≥ 2.4 with a CUDA build matching your driver (see
Troubleshooting below).

### Full SMCEvolve run

```bash
python -m smcevolve.main problem=autoresearch algo=small
```

For a longer training budget per candidate, export `AR_TIME_BUDGET`
and bump the Hydra timeout:

```bash
AR_TIME_BUDGET=180 python -m smcevolve.main \
    problem=autoresearch algo=medium \
    problem.timeout=600
```

`AR_TIME_BUDGET` is read by `prepare.py` at import time and becomes
the training cap. Hydra's `problem.timeout` bounds the full
evaluation (training + startup + final bpb eval); keep it ≳
`AR_TIME_BUDGET + 60 s`.

### Pin the run to a specific GPU

The seed creates a single `torch.device("cuda")`. Use
`CUDA_VISIBLE_DEVICES` to steer it to an idle GPU on a shared box:

```bash
CUDA_VISIBLE_DEVICES=0 python -m smcevolve.main problem=autoresearch algo=small
```

## Compute budget guidance

Each candidate spends roughly:

```
cold start   ~15-20 s   (torch import, tokenizer load, model init)
training     AR_TIME_BUDGET (default 60 s)
final eval   ~3-8 s     (262 K tokens of val bpb)
```

So `algo=small` (~8 particles × ~10 iterations) ≈ 80 candidates ×
~90 s ≈ **~2 hours** on a single A5000 with defaults. Scale up via
`AR_TIME_BUDGET` / `algo=medium|large` at your own time cost.

## Troubleshooting

- **`CUDA initialization: The NVIDIA driver on your system is too old`**:
  the base venv may ship a torch wheel built against a newer CUDA
  than your driver (e.g. `torch==2.11+cu130` on a cu124 host). Pin a
  wheel that matches your driver:

  ```bash
  source .venv/bin/activate
  # For a CUDA 12.4 host (RTX A5000 driver 550.x):
  uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124'
  ```

- **`rustbpe` install fails**: requires a Rust toolchain. `uv pip`
  usually grabs a prebuilt wheel; if it falls back to source,
  `rustup default stable` and retry.
- **`No parquet files found. Run prepare.py first.`**: run
  `python problems/autoresearch/prepare.py` once. The val shard
  (#6542) must be present.
- **OOM during training**: the agent proposed a too-big model. The
  evaluator catches it and returns 0.0 — nothing to do.
- **reward = 0.0 on the seed**: run `python problems/autoresearch/initial_program.py`
  directly to see the stack trace (the evaluator swallows it by
  design).
- **FA3 / `kernels` missing**: you don't need it. The seed uses SDPA.
  If an agent proposal tries to import FA3, let it fail (reward 0)
  or install `flash-attn` yourself.

## License

Upstream `prepare.py` and `train.py` are MIT-licensed (karpathy).
The SMCEvolve wrappers in this directory follow the repo license.
