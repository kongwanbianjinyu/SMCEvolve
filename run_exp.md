# SMCEvolve 实验复现指南

按本文件从上到下走一遍，能从零复现 SMCEvolve 在所有 4 大类问题上的实验。
所有命令默认从 **repo 根目录**（`SMCEvolve/`）执行。

---

## 1. 问题盘点

总共 **148** 个已注册问题，分 4 大类：

| 类别           | 数量 | Hydra 名字前缀 | 每个 candidate 的特点                                   | 资源     |
|----------------|------|----------------|---------------------------------------------------------|----------|
| `math`         | 10   | `math_*`       | NumPy 或 JAX；0.5–30 s/候选                             | CPU      |
| `algotune`     | 8    | `algotune_*`   | 以 wall-clock 时间为 reward (`speedup_score`)；**对 CPU 干扰极度敏感** | CPU (串行) |
| `symreg`       | 129  | `symreg_*`     | BFGS 拟合；~1–5 s/候选（量大但每个都快）                | CPU      |
| `autoresearch` | 1    | `autoresearch` | 60+ s GPU 训练 per candidate；**独占 24 GB A5000**      | GPU      |

随时查清单：

```bash
./run_exp.sh problems            # 全部
./run_exp.sh problems symreg     # 只列某一类
```

---

## 2. 准备

### 2.1 安装 uv 并同步 Python 环境

项目用 [uv](https://docs.astral.sh/uv/) 管 venv；先装一次 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在 repo 根目录创建并激活 venv（`uv sync` 会读 `pyproject.toml` + `uv.lock`）：

```bash
uv sync
source .venv/bin/activate
```

之后所有 `python` / `pip` 命令默认走 `.venv`。

### 2.2 配置 LLM API 凭证

```bash
cp .env.example .env
# 编辑 .env：填 OPENAI_API_KEY 和 API_BASE_URL
```

`.env` 内容举例（任意 OpenAI 兼容 endpoint 都行：OpenAI / Azure / LiteLLM / vLLM / Ollama …）：

```
OPENAI_API_KEY=sk-...
API_BASE_URL=https://litellm.cloud.osu.edu
```

`.env` 已 gitignored，不会被提交。

### 2.3 安装各类问题的额外依赖

每类只需做一次。

```bash
# (a) math —— autocorr 三道题用 JAX；其余 math 题用 NumPy 即可
uv pip install jax optax sympy

# (b) symreg —— 物化 129 个任务的数据 + 生成 Hydra config
uv pip install datasets huggingface_hub h5py sympy scipy scikit-learn pyyaml
python problems/symbolic_regression/data_api.py
python problems/symbolic_regression/generate_smc_configs.py
# 之后 configs/problem/symreg_*.yaml 应该有 ~129 个

# (c) algotune
# 在 SMCEvolve 的上一级目录 clone
git clone https://github.com/oripress/AlgoTune ../AlgoTune

# 安装 AlgoTune 本身的 Python 依赖（jax / cvxpy / pulp / pot / numba / scikit-learn ...）
uv pip install -r problems/algotune/requirements.txt

# 8 个 algotune 任务文件夹已经物化在 repo 里（`problems/algotune/<task>/`），无需重新生成。
# 若要重新生成或新增 task，见 [`problems/algotune/README.md`](problems/algotune/README.md)。


# (d) autoresearch —— 拉数据 + 训 tokenizer
uv pip install -r problems/autoresearch/requirements.txt
python problems/autoresearch/prepare.py
# 如果机器是较老的 CUDA 驱动 (cu124)，把 torch 换成 cu124 build：
# uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124


```


### 2.4 验证每类能跑通

各类挑一个最短的题用 `algo=small` 跑一下（每个应该几分钟内结束）：

```bash
./run_exp.sh single math_kissing_number
./run_exp.sh single algotune_affine_transform_2d
./run_exp.sh single symreg_phys_osc_PO0
./run_exp.sh single autoresearch
```

四个都返回 `ok` 之后，再跑大批量。

---

## 4. 跑整个类别

最常用的入口。`run_exp.sh` 会处理好 evaluator 之间的资源干扰、LLM API 并发、输出隔离。默认用small budget(`algo=small`).

```bash
./run_exp.sh category math           # 10 个问题，并行 3
./run_exp.sh category symreg         # 129 个问题，并行 4
./run_exp.sh category algotune       # 8 个问题，强制串行（时间敏感）
AR_GPUS=0,1,2,3 ./run_exp.sh category autoresearch   # 1 个问题，独占 GPU 0
```

**默认并发** 已按类别选好：

| 类别           | 默认 `PARALLEL` | 与 `LLM_CONCURRENCY=4` 组合后最大在途 LLM 请求 |
|----------------|-----------------|------------------------------------------------|
| `math`         | 3               | 12                                             |
| `symreg`       | 4               | 16                                             |
| `algotune`     | 1（强制）       | 4                                              |
| `autoresearch` | 1（强制）       | 4                                              |

`PARALLEL=<n>` 可覆盖，**但 algotune 并发 >1 会让 wall-clock reward 失真，慎改**。

跑全部 4 类（推荐的整夜 sweep，约 4–8 h，`algo=small`）：

```bash
SWEEP_TAG=full_$(date +%Y%m%d) ./run_exp.sh all
```

`all` 内部执行顺序固定为：

```
symreg (CPU, 并行 4) → math (CPU, 并行 3) → algotune (CPU, 串行) → autoresearch (GPU)
```

——这样 CPU/GPU 不会互相抢，algotune 的时间测量也不会被 symreg/math 污染。

实时跟踪：

```bash
tail -f outputs/_sweep/full_$(date +%Y%m%d)/_sweep.log
```

---

## 5. 输出文件布局

每次 sweep 产出两类目录：

```
outputs/
├── _sweep/<TAG>/                          # 本次 sweep 的元信息
│   ├── _sweep.log                         # 全量 start/ok/FAIL 汇总（最重要的总日志）
│   └── <problem>.log                      # 每个问题的 stdout+stderr
│
└── <category>/<problem>/sweep_<TAG>/      # 单个 run 的产出（也是 Hydra run dir）
    ├── events.jsonl                       # SMCEvolve 事件流（喂给 viz/）
    ├── main.log                           # Hydra + SMCEvolve 人读日志
    └── event_logs/
        ├── _final.json                    # 最终最优程序 + 总结
        └── island_<i>/iter_<n>/...        # 每 island 每 iter 的 prompt/response/program
```

`<category>` ∈ `math` | `algotune` | `symreg` | `autoresearch`。
**同一 `SWEEP_TAG` 重跑同一问题会覆盖上次结果**——要保留就换 `SWEEP_TAG`。

可视化：

```bash
./viz.sh                # http://127.0.0.1:5173
./viz.sh --port 8080
./viz.sh --host 0.0.0.0
```

侧边栏会按 `<category>/<problem>/sweep_<TAG>` 树形展开所有 run。

常用排查命令：

```bash
# 列出 sweep 里所有失败的问题
grep -E '^\[.*\] FAIL' outputs/_sweep/<TAG>/_sweep.log

# 抓某个问题的最终分数
grep -hE '"type": *"final"' outputs/<cat>/<problem>/sweep_<TAG>/events.jsonl \
  | python -c 'import json,sys; [print(json.loads(l).get("best_reward")) for l in sys.stdin]'

# 抓某次 sweep 里所有问题的最终分数
for f in outputs/*/sweep_<TAG>/events.jsonl; do
  echo -n "$(dirname "$f" | sed 's|outputs/||'): "
  grep -hE '"type": *"final"' "$f" \
    | python -c 'import json,sys; [print(json.loads(l).get("best_reward")) for l in sys.stdin]'
done
```

---

## 6. 更多命令

### 6.1 跑单个问题

```bash
./run_exp.sh single math_kissing_number
./run_exp.sh single autoresearch                  # 自动绑 GPU 0
./run_exp.sh single symreg_phys_osc_PO0
```

### 6.2 跑一组显式列出的问题

按类别自动分组；algotune / autoresearch 这一组会强制降为串行：

```bash
./run_exp.sh list \
    math_heilbronn_triangle \
    math_kissing_number \
    algotune_fft_convolution \
    autoresearch
# 执行顺序：symreg/math 并行 → algotune 串行 → autoresearch 独占 GPU
```

### 6.3 Dry-run（只打印不执行）

```bash
DRY_RUN=1 ./run_exp.sh all | head -40
```

### 6.4 可调环境变量

| 变量              | 默认           | 说明                                                                  |
|-------------------|----------------|-----------------------------------------------------------------------|
| `ALGO`            | `small`        | Hydra algo preset：`small` / `medium` / `large` / `smc`               |
| `SEED`            | `42`           | 随机种子                                                              |
| `GPU`             | `0`            | 仅 autoresearch 单卡场景的旧入口；如未设 `AR_GPUS` 则用它              |
| `AR_GPUS`         | `$GPU`         | autoresearch 的 GPU 池（逗号分隔物理 GPU ID，如 `"0,1,2,3"`）。Evaluator 把每次评估 round-robin 到一个 GPU 子进程，并发数自动等于池大小 |
| `PARALLEL`        | 按类别         | 覆盖类别默认并发（algotune/autoresearch 慎改）                        |
| `LLM_CONCURRENCY` | `4`            | 每个 run 的 `llm.max_concurrency`；总在途 = `PARALLEL × LLM_CONCURRENCY` |
| `AR_TIME_BUDGET`  | `60`           | autoresearch 每个 candidate 的训练秒数；只对 autoresearch 生效        |
| `SWEEP_TAG`       | `date+time`    | 输出归到 `outputs/<cat>/<problem>/sweep_<TAG>/` 下                    |
| `DRY_RUN`         | `0`            | `1` = 只打印不执行                                                    |

组合示例：

```bash
# medium preset 跑全部
ALGO=medium SWEEP_TAG=medium_$(date +%Y%m%d) ./run_exp.sh all

# autoresearch 用更长训练预算
AR_TIME_BUDGET=180 ./run_exp.sh category autoresearch

# 临时调大 symreg 并发（机器空闲、API 限额高时）
PARALLEL=8 ./run_exp.sh category symreg

# 用 GPU 2 跑 autoresearch
GPU=2 ./run_exp.sh category autoresearch

# autoresearch 同时用 4 张卡：每个 candidate evaluation 仍占 1 GPU，
# 但 SMC 内部最多 4 个 evaluation 并发执行（自动 round-robin 到 GPU 0/1/2/3）
AR_GPUS=0,1,2,3 ./run_exp.sh category autoresearch

# 同上，但只用 GPU 1 和 GPU 3（最多 2 路并发）
AR_GPUS=1,3 ./run_exp.sh category autoresearch
```

> **算法层并发的来源**：autoresearch 的 `algo.particles_per_island` × `algo.n_islands`
> 决定了一个 SMC step 内能同时跑的 evaluation 数。GPU 池只是上限——
> 如果池有 4 个 GPU 但 `particles_per_island=1` 且 `n_islands=1`，那同一时刻
> 实际只跑 1 个评估。要打满 4 卡，建议：
> ```bash
> AR_GPUS=0,1,2,3 ALGO=medium ./run_exp.sh category autoresearch
> # 或显式调大： algo.particles_per_island=4 (走 Hydra 直跑)
> ```

### 6.5 直接用 Hydra 跑（不经 `run_exp.sh`）

`run_exp.sh` 实际是 Hydra 的封装，下面这条等价于
`./run_exp.sh single math_kissing_number`：

```bash
CUDA_VISIBLE_DEVICES= python -m smcevolve.main \
    problem=math_kissing_number \
    algo=small \
    seed=42 \
    llm.max_concurrency=4 \
    hydra.run.dir=outputs/math/math_kissing_number/sweep_manual
```

任何 Hydra 字段都能从命令行覆盖：

```bash
python -m smcevolve.main problem=circle_packing algo=medium \
    algo.max_iterations=30 algo.n_islands=4 seed=7
```

### 6.6 时间 / 开销粗略估计（`algo=small`，6 iter × 1 island × 4 particles × 1 proposal = 24 LLM 调用 / run）

| 类别              | 每个 run 时长 | 串行总时长   | 默认并行下总时长          |
|-------------------|---------------|--------------|---------------------------|
| symreg (129)      | 3–7 min       | 6–15 h       | 1.5–4 h（PARALLEL=4）     |
| math (10)         | 5–20 min      | 50–200 min   | 20–70 min（PARALLEL=3）   |
| algotune (8)      | 5–15 min      | 40–120 min   | 40–120 min（强制串行）    |
| autoresearch (1)  | ~25 min       | ~25 min      | ~25 min                   |
| **合计**          |               | **~10–20 h** | **~4–8 h**                |

`ALGO=medium` 大约是 `small` 的 3–5 倍。LLM 账单按比例增长——cost 流水写在
`outputs/<cat>/<problem>/sweep_<TAG>/events.jsonl` 的 `proposal.proposal_metadata.cost_usd`
和 `proposal.proposal_metadata.cumulative_cost_usd` 字段里。

---

## 7. 中断与恢复

脚本本身没做 resume。若 sweep 中途挂了：

1. 看 `outputs/_sweep/<TAG>/_sweep.log`，找出哪些没 `ok`；
2. 用 `list` 把没成功的那批重跑（**用同一个 `SWEEP_TAG` 会覆盖原产出**）：

```bash
SWEEP_TAG=<原 TAG> ./run_exp.sh list \
    math_erdos_min_overlap \
    symreg_chem_react_CR17
```

---

## 8. 一句话总结

```bash
# overnight 全量 sweep（small preset，约 4–8 h）
SWEEP_TAG=run1 ./run_exp.sh all

# 实时看进度
tail -f outputs/_sweep/run1/_sweep.log

# 看结果
./viz.sh           # 浏览器打开 http://127.0.0.1:5173
```
