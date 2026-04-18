#!/usr/bin/env bash
# SMCEvolve Ablation Study — Circle Packing (9 runs)
# Usage:
#   ./run_ablation.sh A1          # single experiment
#   ./run_ablation.sh groupA      # one group
#   ./run_ablation.sh all         # all 9 runs
#   SEED=123 ./run_ablation.sh A1 # custom seed
set -euo pipefail
cd "$(dirname "$0")"

SEED="${SEED:-42}"

run_exp() {
  local id=$1; shift
  echo "========================================"
  echo "  Experiment: $id  (seed=$SEED)"
  echo "========================================"
  uv run python -m smcevolve.main \
    problem=circle_packing \
    seed="$SEED" \
    "hydra.run.dir=outputs/circle_packing/ablation/${id}_\${now:%Y-%m-%d_%H-%M-%S}" \
    "$@"
}

COMMON="algo.max_iterations=30"

# ── Baseline (= A3 = B2 = C1) ────────────────────────────────────────────────
run_baseline() {
  run_exp baseline $COMMON \
    algo.beta=20 algo.kappa=0.9 \
    algo.n_islands=2 algo.particles_per_island=8 algo.n_proposals=2
}

# ── Group A: β, κ → 迭代次数 ─────────────────────────────────────────────────
run_A1() { run_exp A1 $COMMON algo.beta=5  algo.kappa=0.9 algo.n_islands=2 algo.particles_per_island=8 algo.n_proposals=2; }
run_A2() { run_exp A2 $COMMON algo.beta=40 algo.kappa=0.9 algo.n_islands=2 algo.particles_per_island=8 algo.n_proposals=2; }
run_A3() { run_exp A3 $COMMON algo.beta=20 algo.kappa=0.5 algo.n_islands=2 algo.particles_per_island=8 algo.n_proposals=2; }
run_A4() { run_exp A4 $COMMON algo.beta=20 algo.kappa=1.0 algo.n_islands=2 algo.particles_per_island=8 algo.n_proposals=2; }

# ── Group B: Population (I×P×K=32) ───────────────────────────────────────────
run_B1() { run_exp B1 $COMMON algo.beta=20 algo.kappa=0.9 algo.n_islands=1 algo.particles_per_island=16 algo.n_proposals=2; }
run_B2() { run_exp B2 $COMMON algo.beta=20 algo.kappa=0.9 algo.n_islands=4 algo.particles_per_island=4  algo.n_proposals=2; }
run_B3() { run_exp B3 $COMMON algo.beta=20 algo.kappa=0.9 algo.n_islands=1 algo.particles_per_island=8  algo.n_proposals=4; }
run_B4() { run_exp B4 $COMMON algo.beta=20 algo.kappa=0.9 algo.n_islands=2 algo.particles_per_island=16 algo.n_proposals=1; }

# ── Group C: Kernel ───────────────────────────────────────────────────────────
COMMON_C="$COMMON algo.beta=20 algo.kappa=0.9 algo.n_islands=2 algo.particles_per_island=8 algo.n_proposals=2"
run_C1() { run_exp C1 $COMMON_C +algo.prompt.force_kernel=diff_no_inspo; }
run_C2() { run_exp C2 $COMMON_C +algo.prompt.force_kernel=diff_with_inspo; }
run_C3() { run_exp C3 $COMMON_C +algo.prompt.force_kernel=rewrite_no_inspo; }
run_C4() { run_exp C4 $COMMON_C +algo.prompt.force_kernel=rewrite_with_inspo; }

# ── Group runners ─────────────────────────────────────────────────────────────
run_groupA() { run_A1; run_A2; run_A3; run_A4; }
run_groupB() { run_B1; run_B2; run_B3; run_B4; }
run_groupC() { run_C1; run_C2; run_C3; run_C4; }
run_all()    { run_A1; run_A2; run_A3; run_A4; run_B1; run_B2; run_B3; run_B4; run_C1; run_C2; run_C3; run_C4; }

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${1:-}" in
  baseline)  run_baseline ;;
  A[1-4])    "run_$1" ;;
  B[1-4])    "run_$1" ;;
  C[1-4])    "run_$1" ;;
  groupA)    run_groupA ;;
  groupB)    run_groupB ;;
  groupC)    run_groupC ;;
  all)       run_all ;;
  *)
    echo "Usage: $0 <experiment|group|all>"
    echo ""
    echo "  baseline          β=20 κ=0.9 I=2 P=8 K=2 adaptive"
    echo "  A1  β=5   κ=0.9  A2  β=40  κ=0.9  A3  β=20  κ=0.5  A4  β=20  κ=1.0"
    echo "  B1 (1,16,2)       B2 (4,4,2)        B3 (1,8,4)"
    echo "  C1  diff+inspo    C2  diff only"
    echo "  groupA groupB groupC all"
    exit 1 ;;
esac
