#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
RUNS="${RUNS:-1}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ROOT_OUTDIR="${ROOT_OUTDIR:-experiments}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

STUDY="${1:-all}"   # models | losses | obs_window | all
case "$STUDY" in
  models|losses|obs_window|all) ;;
  *)
    echo "Usage: $0 [models|losses|obs_window|all]"
    echo "Env overrides: RUNS, EPOCHS, BATCH_SIZE, ROOT_OUTDIR, PYTHON_BIN, DRY_RUN"
    exit 1
    ;;
esac

timestamp() { date +"%Y%m%d_%H%M%S"; }

run_one() {
  local tag="$1" model="$2" loss="$3" cluster="$4" obs_window="${5:-}"

  local ts outdir logdir
  ts="$(timestamp)"
  outdir="${ROOT_OUTDIR}/${tag}/${ts}"
  logdir="${outdir}/logs"
  mkdir -p "$logdir"

  cat > "${outdir}/TITLE.txt" <<EOF
Experiment: ${tag}
Model: ${model}
Loss: ${loss}
Clustering: ${cluster}
Obs window: ${obs_window}
Runs: ${RUNS}
Epochs: ${EPOCHS}
Batch size: ${BATCH_SIZE}
Started at: ${ts}
EOF

  # Write machine-readable config (obs_window may be null)
  cat > "${outdir}/config.json" <<EOF
{"tag":"${tag}","model":"${model}","loss":"${loss}","clustering":"${cluster}","obs_window":${obs_window:-null},"runs":${RUNS},"epochs":${EPOCHS},"batch_size":${BATCH_SIZE},"started_at":"${ts}"}
EOF

  echo ">>> Running ${tag} | model=${model} | loss=${loss} | clustering=${cluster} | obs_window=${obs_window:-} | runs=${RUNS}"

  # training command
  cmd=( "$PYTHON_BIN" train_models.py
        --selected_model "${model}"
        --selected_loss_function "${loss}"
        --num_epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --clustering "${cluster}" )

  # only add obs_window flag if provided
  if [ -n "${obs_window:-}" ]; then
    cmd+=( --obs_window "${obs_window}" )
  fi

  for i in $(seq 1 "${RUNS}"); do
    echo "=== ${tag} — run ${i}/${RUNS} ==="
    if [ "$DRY_RUN" = "1" ]; then
      echo "(dry-run) ${cmd[*]} | tee ${logdir}/run_${i}.log"
    else
      "${cmd[@]}" 2>&1 | tee "${logdir}/run_${i}.log"
    fi
  done

  if [ "$DRY_RUN" = "0" ]; then
    "$PYTHON_BIN" - "$outdir" <<'PY'
import csv, json, ast, numpy as np, pathlib, sys
outdir = pathlib.Path(sys.argv[1])
logdir = outdir / "logs"
logs = sorted(logdir.glob("run_*.log"))

runs=[]
for log in logs:
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start = None
    for i, l in enumerate(lines):
        if l.strip() == "Test set metrics:":
            start = i + 1
            break
    if start is None:
        continue
    metrics = {}
    for l in lines[start:]:
        if not l.startswith("  "):
            if l.strip() == "":
                break
            continue
        s = l.strip()
        if ":" not in s:
            continue
        key, val = s.split(":", 1)
        key = key.strip(); val = val.strip()
        parsed = None
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

with open(outdir/"per_run_metrics.json", "w") as f:
    json.dump(runs, f, indent=2)

keys = sorted({k for r in runs for k in r})
summary_rows = []
for k in keys:
    vals = [r[k] for r in runs if k in r]
    if not vals: continue
    if isinstance(vals[0], (int, float)):
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary_rows.append([k, mean, std]); continue
    if isinstance(vals[0], list) and (not isinstance(vals[0][0], list)):
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean(axis=0).tolist()
        std = (arr.std(axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [0.0]*arr.shape[1])
        summary_rows.append([k, json.dumps(mean), json.dumps(std)]); continue
    if isinstance(vals[0], list) and isinstance(vals[0][0], list):
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean(axis=0).tolist()
        std = (arr.std(axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [[0.0]*len(arr[0][0])]*len(arr[0]))
        summary_rows.append([k, json.dumps(mean), json.dumps(std)]); continue
    summary_rows.append([k, json.dumps(vals), ""])

csv_path = outdir / "metrics_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric","mean","std"])
    for row in summary_rows: w.writerow(row)

print(f"Wrote summary CSV to: {csv_path}")
PY
  fi

  mkdir -p "${ROOT_OUTDIR}"
  local index="${ROOT_OUTDIR}/_INDEX.csv"
  if [ ! -f "$index" ]; then
    echo "timestamp,tag,model,loss,clustering,obs_window,runs,epochs,batch_size,outdir" > "$index"
  fi
  echo "$(timestamp),${tag},${model},${loss},${cluster},${obs_window},${RUNS},${EPOCHS},${BATCH_SIZE},${outdir}" >> "$index"
}

# -----------------------------
# Experiment specs
# -----------------------------
read -r -d '' EXP_MODELS <<'EOF' || true
tag=models_ablation__parametric_gtcnn_event__dice__clustered model=parametric_gtcnn_event loss=dice cluster=True obs_window=6
tag=models_ablation__parametric_gtcnn__dice__clustered       model=parametric_gtcnn       loss=dice cluster=True obs_window=6
tag=models_ablation__vanilla_gcnn__dice__fullbatch           model=vanilla_gcnn           loss=dice              obs_window=4
tag=models_ablation__simple_gc__dice__fullbatch              model=simple_gc              loss=dice              obs_window=4
EOF

read -r -d '' EXP_LOSSES <<'EOF' || true
tag=losses_ablation__parametric_gtcnn_event__focal__clustered        model=parametric_gtcnn_event loss=focal        cluster=True obs_window=6
tag=losses_ablation__parametric_gtcnn_event__dice__clustered         model=parametric_gtcnn_event loss=dice         cluster=True obs_window=6
tag=losses_ablation__parametric_gtcnn_event__weighted_bce__clustered model=parametric_gtcnn_event loss=weighted_bce cluster=True obs_window=6
tag=losses_ablation__parametric_gtcnn_event__bce__clustered          model=parametric_gtcnn_event loss=bce          cluster=True obs_window=6
EOF

# obs_window tuning experiments
read -r -d '' EXP_OBS_WINDOW <<'EOF' || true
tag=models_ablation__parametric_gtcnn_event__focal__clustered__ow1 model=parametric_gtcnn_event loss=focal cluster=True obs_window=1
tag=models_ablation__parametric_gtcnn_event__focal__clustered__ow4 model=parametric_gtcnn_event loss=focal cluster=True obs_window=4
tag=models_ablation__parametric_gtcnn_event__focal__clustered__ow6 model=parametric_gtcnn_event loss=focal cluster=True obs_window=6
EOF

run_group() {
  local list="$1"
  local line k v tag model loss cluster obs_window
  while IFS= read -r line; do
    [ -z "${line// }" ] && continue
    tag=""; model=""; loss=""; cluster=""; obs_window=""
    for kv in $line; do
      k="${kv%%=*}"; v="${kv#*=}"
      case "$k" in
        tag) tag="$v" ;;
        model) model="$v" ;;
        loss) loss="$v" ;;
        cluster) cluster="$v" ;;
        obs_window) obs_window="$v" ;;
      esac
    done
    run_one "$tag" "$model" "$loss" "$cluster" "$obs_window"
  done <<EOF
$list
EOF
}

case "$STUDY" in
  models)         run_group "$EXP_MODELS" ;;
  losses)         run_group "$EXP_LOSSES" ;;
  obs_window)     run_group "$EXP_OBS_WINDOW" ;;
  all)            run_group "$EXP_MODELS"; run_group "$EXP_LOSSES"; run_group "$EXP_OBS_WINDOW" ;;
esac

echo "Done."
