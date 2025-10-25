#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (you can tweak below)
# -----------------------------
RUNS="${RUNS:-3}"                # how many repeats per experiment
EPOCHS="${EPOCHS:-50}"           # num_epochs flag
BATCH_SIZE="${BATCH_SIZE:-16}"   # batch_size flag
ROOT_OUTDIR="${ROOT_OUTDIR:-experiments}"
PYTHON_BIN="${PYTHON_BIN:-python3}"  # python executable
DRY_RUN="${DRY_RUN:-0}"          # set to 1 to only print commands

# -----------------------------
# Parse which study to run
# -----------------------------
STUDY="${1:-all}"   # one of: models | losses | all
case "$STUDY" in
  models|losses|all) ;;
  *)
    echo "Usage: $0 [models|losses|all]"
    echo "Env overrides: RUNS, EPOCHS, BATCH_SIZE, ROOT_OUTDIR, PYTHON_BIN, DRY_RUN"
    exit 1
    ;;
esac

timestamp() { date +"%Y%m%d_%H%M%S"; }

# -----------------------------
# Helper to run one experiment
# -----------------------------
run_one() {
  local tag="$1"       # human-friendly tag for folder and title
  local model="$2"
  local loss="$3"
  local cluster="$4"   # True or False

  local ts; ts="$(timestamp)"
  local outdir="${ROOT_OUTDIR}/${tag}/${ts}"
  local logdir="${outdir}/logs"
  mkdir -p "$logdir"

  # Explainable title + config snapshot
  cat > "${outdir}/TITLE.txt" <<EOF
Experiment: ${tag}
Model: ${model}
Loss: ${loss}
Clustering: ${cluster}
Runs: ${RUNS}
Epochs: ${EPOCHS}
Batch size: ${BATCH_SIZE}
Started at: ${ts}
EOF

  cat > "${outdir}/config.json" <<EOF
{"tag":"${tag}","model":"${model}","loss":"${loss}","clustering":"${cluster}","runs":${RUNS},"epochs":${EPOCHS},"batch_size":${BATCH_SIZE},"started_at":"${ts}"}
EOF

  echo ">>> Running ${tag} | model=${model} | loss=${loss} | clustering=${cluster} | runs=${RUNS}"

  # Training command
  # Note: clustering must be passed as a Pythonic boolean literal (True/False)
  cmd=( "${PYTHON_BIN}" train_models.py
        --selected_model "${model}"
        --selected_loss_function "${loss}"
        --num_epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --clustering "${cluster}" )

  # Repeat runs and log
  for i in $(seq 1 "${RUNS}"); do
    echo "=== ${tag} — run ${i}/${RUNS} ==="
    if [ "$DRY_RUN" = "1" ]; then
      echo "(dry-run) ${cmd[*]} | tee ${logdir}/run_${i}.log"
    else
      "${cmd[@]}" 2>&1 | tee "${logdir}/run_${i}.log"
    fi
  done

  # Aggregate metrics (mean & std) into CSV
  if [ "$DRY_RUN" = "0" ]; then
    "${PYTHON_BIN}" - << 'PY' "${outdir}"
import csv, json, ast, numpy as np, pathlib, sys
outdir = pathlib.Path(sys.argv[1])
logdir = outdir / "logs"
logs = sorted(logdir.glob("run_*.log"))

runs=[]
for log in logs:
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # find the "Test set metrics:" section
    start = None
    for i, l in enumerate(lines):
        if l.strip() == "Test set metrics:":
            start = i + 1
            break
    if start is None:
        continue
    metrics = {}
    for l in lines[start:]:
        # Only accept "  key: value" lines (two-space indent like in train_models.py)
        if not l.startswith("  "):
            # stop when leaving the metrics block
            if l.strip() == "":
                break
            continue
        s = l.strip()
        if ":" not in s:
            continue
        key, val = s.split(":", 1)
        key = key.strip()
        val = val.strip()
        parsed = None
        # Try to parse as Python literal, then float
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            try:
                parsed = float(val)
            except Exception:
                pass
        if parsed is not None:
            metrics[key] = parsed
    if metrics:
        runs.append(metrics)

if not runs:
    raise SystemExit("No metrics parsed. Ensure the logs contain 'Test set metrics:' with '  key: value' lines.")

# Persist per-run metrics for auditing
with open(outdir/"per_run_metrics.json", "w") as f:
    json.dump(runs, f, indent=2)

# All metric keys
keys = sorted({k for r in runs for k in r})
summary_rows = []
for k in keys:
    vals = [r[k] for r in runs if k in r]
    if not vals:
        continue
    # Scalars
    if isinstance(vals[0], (int, float)):
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary_rows.append([k, mean, std]); continue
    # 1D lists
    if isinstance(vals[0], list) and (not isinstance(vals[0][0], list)):
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean(axis=0).tolist()
        std = (arr.std(axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [0.0]*arr.shape[1])
        summary_rows.append([k, json.dumps(mean), json.dumps(std)]); continue
    # 2D lists (e.g., confusion_matrix)
    if isinstance(vals[0], list) and isinstance(vals[0][0], list):
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean(axis=0).tolist()
        std = (arr.std(axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [[0.0]*len(arr[0][0])]*len(arr[0]))
        summary_rows.append([k, json.dumps(mean), json.dumps(std)]); continue
    # Fallback
    summary_rows.append([k, json.dumps(vals), ""])

csv_path = outdir / "metrics_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric","mean","std"])
    for row in summary_rows: w.writerow(row)

print(f"Wrote summary CSV to: {csv_path}")
PY
  fi

  # Append to global index
  mkdir -p "${ROOT_OUTDIR}"
  local index="${ROOT_OUTDIR}/_INDEX.csv"
  if [ ! -f "$index" ]; then
    echo "timestamp,tag,model,loss,clustering,runs,epochs,batch_size,outdir" > "$index"
  fi
  echo "$(timestamp),${tag},${model},${loss},${cluster},${RUNS},${EPOCHS},${BATCH_SIZE},${outdir}" >> "$index"
}

# -----------------------------
# Define experiment sets
# -----------------------------
# 1) Models ablation (fixed loss=focal)
declare -a EXP_MODELS=(
  "tag=models_ablation__parametric_gtcnn_event__focal__clustered model=parametric_gtcnn_event loss=focal cluster=True"
  "tag=models_ablation__parametric_gtcnn__focal__clustered       model=parametric_gtcnn       loss=focal cluster=True"
  "tag=models_ablation__vanilla_gcnn__focal__fullbatch           model=vanilla_gcnn           loss=focal cluster=False"
)

# 2) Losses ablation (fixed model=parametric_gtcnn_event)
declare -a EXP_LOSSES=(
  "tag=losses_ablation__parametric_gtcnn_event__dice__clustered         model=parametric_gtcnn_event loss=dice         cluster=True"
  "tag=losses_ablation__parametric_gtcnn_event__weighted_bce__clustered model=parametric_gtcnn_event loss=weighted_bce cluster=True"
  "tag=losses_ablation__parametric_gtcnn_event__bce__clustered          model=parametric_gtcnn_event loss=bce          cluster=True"
)

# -----------------------------
# Execute selected study
# -----------------------------
run_group() {
  local -n arr="$1"
  for spec in "${arr[@]}"; do
    # parse spec into variables
    local tag model loss cluster
    tag=""; model=""; loss=""; cluster=""
    for kv in $spec; do
      k="${kv%%=*}"; v="${kv#*=}"
      case "$k" in
        tag) tag="$v" ;;
        model) model="$v" ;;
        loss) loss="$v" ;;
        cluster) cluster="$v" ;;
      esac
    done
    run_one "$tag" "$model" "$loss" "$cluster"
  done
}

case "$STUDY" in
  models) run_group EXP_MODELS ;;
  losses) run_group EXP_LOSSES ;;
  all)    run_group EXP_MODELS; run_group EXP_LOSSES ;;
esac

echo "Done."
