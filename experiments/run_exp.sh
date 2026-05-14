#!/usr/bin/env bash
# run_exp.sh — SMCEvolve experiment dispatcher.
# See run_exp.md (alongside this script) for the design rationale and recipes.
set -euo pipefail
# This script lives in experiments/ but operates on the repo root
# (configs/, outputs/, problems/ all live there).
cd "$(dirname "$0")/.."

# -------- config (env-overrideable) --------
: "${ALGO:=small}"
: "${SEED:=42}"
: "${GPU:=0}"
: "${LLM_CONCURRENCY:=4}"
: "${PARALLEL:=}"            # empty = use per-category default
: "${AR_TIME_BUDGET:=60}"
: "${AR_GPUS:=$GPU}"         # comma-separated physical GPU IDs for autoresearch
: "${SWEEP_TAG:=$(date +%Y%m%d_%H%M)}"
: "${DRY_RUN:=0}"

# -------- problem catalogs --------
MATH=(
  math_heilbronn_triangle
  math_circle_packing_rect
  math_kissing_number
  math_hexagon_packing_11
  math_heilbronn_convex_13
  math_minimizing_max_min_dist_dim2_16
  math_first_autocorr_ineq
  math_second_autocorr_ineq
  math_third_autocorr_ineq
  math_erdos_min_overlap
)
ALGOTUNE=(
  algotune_affine_transform_2d
  algotune_convolve2d_full_fill
  algotune_eigenvectors_complex
  algotune_fft_cmplx_scipy_fftpack
  algotune_fft_convolution
  algotune_lu_factorization
  algotune_polynomial_real
  algotune_psd_cone_projection
)
AUTORESEARCH=(autoresearch)

# symreg is materialized dynamically — there are ~129 problems
if compgen -G "configs/problem/symreg_*.yaml" > /dev/null; then
  mapfile -t SYMREG < <(ls configs/problem/symreg_*.yaml | xargs -n1 basename -s .yaml)
else
  SYMREG=()
fi

# -------- per-category safe parallelism --------
default_parallel() {
  case "$1" in
    math)         echo 3 ;;
    symreg)       echo 4 ;;
    algotune)     echo 1 ;;   # timing-sensitive: serial only
    autoresearch) echo 1 ;;   # single-GPU: serial only
    *)            echo 1 ;;
  esac
}

# -------- sweep logging --------
SWEEPDIR="outputs/_sweep/${SWEEP_TAG}"
mkdir -p "$SWEEPDIR"
SWEEP_LOG="$SWEEPDIR/_sweep.log"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$SWEEP_LOG"; }

category_of() {
  case "$1" in
    math_*)          echo math ;;
    algotune_*)      echo algotune ;;
    symreg_*)        echo symreg ;;
    autoresearch)    echo autoresearch ;;
    *)               echo other ;;
  esac
}

# -------- run one problem --------
run_one() {
  local problem="$1"
  local cat; cat=$(category_of "$problem")
  local logfile="$SWEEPDIR/${problem}.log"

  local env_prefix=()
  if [[ "$cat" == "autoresearch" ]]; then
    # Pin parent's CUDA_VISIBLE_DEVICES to AR_GPUS so the GPUs the user picked
    # become parent-visible indices 0..N-1. The Evaluator GPU pool then dishes
    # out those parent-visible indices to subprocess CUDA_VISIBLE_DEVICES,
    # which is portable regardless of which physical GPU IDs were chosen.
    local _gpu_arr; IFS=',' read -ra _gpu_arr <<< "$AR_GPUS"
    local _n=${#_gpu_arr[@]}
    local _indices=""
    local _i
    for ((_i=0; _i<_n; _i++)); do _indices+="$_i,"; done
    _indices="${_indices%,}"
    env_prefix=(env
      "CUDA_VISIBLE_DEVICES=$AR_GPUS"
      "EVAL_GPU_IDS=$_indices"
      "AR_TIME_BUDGET=$AR_TIME_BUDGET"
    )
  else
    # CPU-only: prevents GPU collisions with autoresearch and avoids
    # JAX grabbing the GPU for the math autocorr problems.
    env_prefix=(env "CUDA_VISIBLE_DEVICES=")
  fi

  local cmd=(
    "${env_prefix[@]}"
    python -m smcevolve.main
    "problem=$problem"
    "algo=$ALGO"
    "seed=$SEED"
    "llm.max_concurrency=$LLM_CONCURRENCY"
    "hydra.run.dir=outputs/${cat}/${problem}/sweep_${SWEEP_TAG}"
  )

  log "start  $problem"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY    ${cmd[*]}"
    return 0
  fi
  local t0 t1
  t0=$(date +%s)
  if "${cmd[@]}" > "$logfile" 2>&1; then
    t1=$(date +%s)
    log "ok     $problem ($((t1-t0))s)"
  else
    t1=$(date +%s)
    log "FAIL   $problem ($((t1-t0))s)  see $logfile"
  fi
}

# -------- bounded-parallel dispatcher --------
run_list() {
  local cat="$1"; shift
  local -a problems=("$@")
  local total=${#problems[@]}
  if (( total == 0 )); then
    log "skip   category=$cat (no problems)"
    return 0
  fi
  local par="${PARALLEL:-$(default_parallel "$cat")}"
  log "==== category=$cat  n=$total  parallel=$par  algo=$ALGO ===="
  local i=0
  for p in "${problems[@]}"; do
    while (( $(jobs -rp | wc -l) >= par )); do sleep 1; done
    i=$((i+1))
    log "launch [$i/$total] $p"
    run_one "$p" &
  done
  wait
  log "==== done  category=$cat ===="
}

# -------- dispatch --------
cmd="${1:-}"
shift || true

show_usage() {
  cat <<EOF
Usage: $0 <cmd> [args]

Commands:
  single <problem>          Run a single problem (Hydra name, e.g. math_kissing_number)
  list <problem> ...        Run an explicit list (grouped by category so algotune/autoresearch stay serial)
  category <cat>            cat = math | symreg | algotune | autoresearch
  all                       Run every registered problem in the safest order
  problems [cat]            List available problems (optionally filtered by category)

Env overrides (defaults shown):
  ALGO=small                small | medium | large
  SEED=42
  GPU=0                     (autoresearch single-GPU default; superseded by AR_GPUS)
  AR_GPUS=\$GPU              comma-separated physical GPU IDs (e.g. "0,1,2,3") — autoresearch
                            evaluations are dispatched round-robin across this pool, so up
                            to len(AR_GPUS) candidates can be evaluated concurrently
  PARALLEL=                 empty = per-category default (math=3, symreg=4, algotune=1, autoresearch=1)
  LLM_CONCURRENCY=4         per-run cap; total in-flight = PARALLEL * LLM_CONCURRENCY
  AR_TIME_BUDGET=60         seconds of training per autoresearch candidate
  SWEEP_TAG=<date+time>     groups outputs under outputs/<problem>/sweep_<TAG>/
  DRY_RUN=0                 1 = print commands, do nothing
EOF
}

# Record sweep wall-clock for runtime commands (skip the read-only `problems`).
fmt_elapsed() {
  local s=$1
  printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}
case "$cmd" in
  single|list|category|all)
    SWEEP_T0=$(date +%s)
    log "==== sweep start  cmd=$cmd $*  algo=$ALGO  seed=$SEED  tag=$SWEEP_TAG ===="
    trap '_rc=$?; _t1=$(date +%s); _el=$((_t1 - SWEEP_T0)); log "==== sweep done   elapsed=$(fmt_elapsed $_el) (${_el}s)  exit=${_rc} ===="' EXIT
    ;;
esac

case "$cmd" in
  single)
    [[ -z "${1:-}" ]] && { show_usage; exit 1; }
    run_list "$(category_of "$1")" "$1"
    ;;
  list)
    [[ $# -eq 0 ]] && { show_usage; exit 1; }
    # Group by category so algotune / autoresearch can stay serial.
    declare -A buckets=()
    for p in "$@"; do
      c=$(category_of "$p")
      buckets[$c]+="$p "
    done
    # Safe execution order: fast CPU first, timing-sensitive + GPU last.
    for c in symreg math algotune autoresearch other; do
      if [[ -n "${buckets[$c]:-}" ]]; then
        # shellcheck disable=SC2086
        run_list "$c" ${buckets[$c]}
      fi
    done
    ;;
  category)
    case "${1:-}" in
      math)         run_list math "${MATH[@]}" ;;
      symreg)       run_list symreg "${SYMREG[@]}" ;;
      algotune)     run_list algotune "${ALGOTUNE[@]}" ;;
      autoresearch) run_list autoresearch "${AUTORESEARCH[@]}" ;;
      *) show_usage; exit 1 ;;
    esac
    ;;
  all)
    # Order chosen for safety + throughput:
    # 1. symreg  (CPU, many problems, dominates count — get it started)
    # 2. math    (CPU; can overlap with nothing that shares CPU w/ symreg)
    # 3. algotune (serial, timing-sensitive — NO other CPU-heavy overlap)
    # 4. autoresearch (GPU alone; anything earlier was CPU-only)
    run_list symreg       "${SYMREG[@]}"
    run_list math         "${MATH[@]}"
    run_list algotune     "${ALGOTUNE[@]}"
    run_list autoresearch "${AUTORESEARCH[@]}"
    ;;
  problems)
    case "${1:-}" in
      math)         printf '%s\n' "${MATH[@]}" ;;
      symreg)       printf '%s\n' "${SYMREG[@]}" ;;
      algotune)     printf '%s\n' "${ALGOTUNE[@]}" ;;
      autoresearch) printf '%s\n' "${AUTORESEARCH[@]}" ;;
      '' )
        printf '# math (%d)\n'         "${#MATH[@]}";         printf '%s\n' "${MATH[@]}"
        printf '# algotune (%d)\n'     "${#ALGOTUNE[@]}";     printf '%s\n' "${ALGOTUNE[@]}"
        printf '# symreg (%d)\n'       "${#SYMREG[@]}";       printf '%s\n' "${SYMREG[@]}"
        printf '# autoresearch (%d)\n' "${#AUTORESEARCH[@]}"; printf '%s\n' "${AUTORESEARCH[@]}"
        ;;
      *) show_usage; exit 1 ;;
    esac
    ;;
  *)
    show_usage
    exit 1 ;;
esac
