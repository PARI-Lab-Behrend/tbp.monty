#!/usr/bin/env bash
#
# run_lbp_comparison.sh — LBP vs LTP vs control (HSV-only) on 11retextured.
#
# Answers the reviewer question "why LTP rather than plain LBP?" by running the
# same three-arm comparison under both lighting conditions:
#
#   noltp : HSV only                              (control, current Monty default)
#   ltp   : HSV + split LTP, 'ror', tol 0.50      (the paper's best variant)
#   lbp   : HSV + classic LBP, 'ror', tol 0.50    (new control descriptor)
#
# The LBP arm needs its own pretraining because the stored texture histogram
# differs (36 'ror' bins vs 2 x 36 for split LTP); everything else — sampling
# geometry, encoding, feature weight, tolerance, policy, seed — is held fixed, so
# the arms differ only in the descriptor. All arms use seed 42 and the same
# RandomRotation env interface, so episodes pair one-to-one across arms and the
# comparison is analysed with paired tests (McNemar on accuracy, paired
# permutation / Wilcoxon on matching steps).
#
# n = 220 episodes (20 rotations x 11 objects) ambient, 110 (10 x 11) directional.
#
# Structure and scheduler are inherited from run_amb_pvalue_study.sh: at most
# MAX_SLOTS serial run.py processes in flight, idempotent/resumable (trainings skip
# when model.pt exists, evals skip when their snapshot exists).
#
# Prerequisites:
#   conda activate tbp.monty
#
# Usage:
#   scripts/run_lbp_comparison.sh            # train + eval + analyze
#   scripts/run_lbp_comparison.sh analyze    # only (re)run the paired analysis
#
set -u -o pipefail

# --------------------------- configuration ---------------------------------
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$REPO"                                     # repo root holds the run.py wrapper
                                                    # (calls setup_env + main)
PY="${PY:-python}"                                  # expects the tbp.monty env
MAX_SLOTS="${MAX_SLOTS:-2}"                         # concurrent (serial) run.py processes
POLL_SECS="${POLL_SECS:-20}"

# Fresh output base for evals (never shared with the messy monty_runs history) and
# an immutable snapshot dir. Both are created if missing.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${MONTY_LOGS:-$HOME/tbp/results/monty}/projects/monty_runs_lbpcmp}"
SNAP_DIR="${SNAP_DIR:-$EVAL_OUT_DIR/_snapshots}"
LOG_DIR="$SNAP_DIR/logs"
SCHED_DIR="$SNAP_DIR/.sched"

MODELS_ROOT="${MONTY_MODELS:-$HOME/tbp/results/monty/pretrained_models}/my_trained_models"

# --------------------------- job table -------------------------------------
# id | kind(train|eval) | hydra experiment config | run_name | deps(comma or -)
# run_name for train == model dir name; for eval == output subdir + snapshot key.
# Order = launch priority (long-pole 77obj training first).
JOBS=(
  "t_amb|train|lbpcmp_amb_train_11retex_lbp|amb_surf_pre_training_11retextured_obj_lbp|-"
  "t_dir|train|lbpcmp_dir_train_11retex_lbp|surf_pre_training_11retextured_obj_lbp|-"
  # Control and LTP arms reuse the existing LTP-pretrained models (the training
  # learning module never uses the texture feature to build graphs, so the graphs
  # are the same; the non-texture LM simply ignores the stored histogram).
  "e_amb_noltp|eval|lbpcmp_amb_eval_11retex_noltp|lbpcmp_amb_eval_11retex_noltp|-"
  "e_amb_ltp|eval|lbpcmp_amb_eval_11retex_ltp|lbpcmp_amb_eval_11retex_ltp|-"
  "e_amb_lbp|eval|lbpcmp_amb_eval_11retex_lbp|lbpcmp_amb_eval_11retex_lbp|t_amb"
  "e_dir_noltp|eval|lbpcmp_dir_eval_11retex_noltp|lbpcmp_dir_eval_11retex_noltp|-"
  "e_dir_ltp|eval|lbpcmp_dir_eval_11retex_ltp|lbpcmp_dir_eval_11retex_ltp|-"
  "e_dir_lbp|eval|lbpcmp_dir_eval_11retex_lbp|lbpcmp_dir_eval_11retex_lbp|t_dir"
  # LBP evaluated at its OWN gap-maximizing tolerance (0.40, from the Hellinger
  # distance analysis on the LBP model) rather than LTP's 0.50, so the baseline
  # cannot be accused of being handicapped by an LTP-tuned threshold.
  "e_amb_lbp040|eval|lbpcmp_amb_eval_11retex_lbp_t040|lbpcmp_amb_eval_11retex_lbp_t040|t_amb"
  "e_dir_lbp040|eval|lbpcmp_dir_eval_11retex_lbp_t040|lbpcmp_dir_eval_11retex_lbp_t040|t_dir"
)

# Paired comparisons to run at the end: "name|label_a:run_a|label_b:run_b"
PAIRS=(
  # Each descriptor against the HSV-only control...
  "amb_ltp_vs_ctrl|LTP:lbpcmp_amb_eval_11retex_ltp|HSV:lbpcmp_amb_eval_11retex_noltp"
  "amb_lbp_vs_ctrl|LBP:lbpcmp_amb_eval_11retex_lbp|HSV:lbpcmp_amb_eval_11retex_noltp"
  "dir_ltp_vs_ctrl|LTP:lbpcmp_dir_eval_11retex_ltp|HSV:lbpcmp_dir_eval_11retex_noltp"
  "dir_lbp_vs_ctrl|LBP:lbpcmp_dir_eval_11retex_lbp|HSV:lbpcmp_dir_eval_11retex_noltp"
  # ...and head to head, which is the reviewer's actual question.
  "amb_ltp_vs_lbp|LTP:lbpcmp_amb_eval_11retex_ltp|LBP:lbpcmp_amb_eval_11retex_lbp"
  "dir_ltp_vs_lbp|LTP:lbpcmp_dir_eval_11retex_ltp|LBP:lbpcmp_dir_eval_11retex_lbp"
  # Same, with LBP at its own tuned tolerance.
  "amb_lbp040_vs_ctrl|LBP.40:lbpcmp_amb_eval_11retex_lbp_t040|HSV:lbpcmp_amb_eval_11retex_noltp"
  "dir_lbp040_vs_ctrl|LBP.40:lbpcmp_dir_eval_11retex_lbp_t040|HSV:lbpcmp_dir_eval_11retex_noltp"
  "amb_ltp_vs_lbp040|LTP:lbpcmp_amb_eval_11retex_ltp|LBP.40:lbpcmp_amb_eval_11retex_lbp_t040"
  "dir_ltp_vs_lbp040|LTP:lbpcmp_dir_eval_11retex_ltp|LBP.40:lbpcmp_dir_eval_11retex_lbp_t040"
)

# --------------------------- helpers ---------------------------------------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

field() { echo "$1" | cut -d'|' -f"$2"; }

model_exists() { [ -f "$MODELS_ROOT/$1/pretrained/model.pt" ]; }
snap_path()    { echo "$SNAP_DIR/eval_stats_$1.csv"; }

require_env() {
  # The run.py wrapper calls setup_env(), which defaults MONTY_MODELS/
  # MONTY_LOGS/MONTY_DATA to ~/tbp/... when unset, so we don't hard-require them --
  # we just report the effective paths this script will use for reuse checks and
  # snapshots (must match where the runs actually read/write).
  if ! "$PY" -c 'import tbp.monty' 2>/dev/null; then
    log "ERROR: '$PY' cannot import tbp.monty (run 'conda activate tbp.monty' or set PY)."
    exit 1
  fi
  log "models root (reuse checks): $MODELS_ROOT"
  log "eval output base:           $EVAL_OUT_DIR"
}

# Run one job (called in the background). Writes .done on success, .failed on error.
run_job() {
  local spec="$1"
  local id kind config run_name
  id="$(field "$spec" 1)"; kind="$(field "$spec" 2)"
  config="$(field "$spec" 3)"; run_name="$(field "$spec" 4)"
  local logf="$LOG_DIR/$id-$run_name.log"

  {
    if [ "$kind" = "train" ]; then
      if model_exists "$run_name"; then
        log "SKIP  $id ($run_name): model already exists — reusing."
        touch "$SCHED_DIR/$id.done"; exit 0
      fi
      log "TRAIN $id -> $run_name"
      ( cd "$RUN_DIR" && "$PY" run.py experiment="$config" ) >"$logf" 2>&1
    else
      if [ -f "$(snap_path "$run_name")" ]; then
        log "SKIP  $id ($run_name): snapshot already exists."
        touch "$SCHED_DIR/$id.done"; exit 0
      fi
      log "EVAL  $id -> $run_name"
      ( cd "$RUN_DIR" && "$PY" run.py experiment="$config" \
          experiment.config.logging.output_dir="$EVAL_OUT_DIR" ) >"$logf" 2>&1
    fi
    rc=$?
    if [ $rc -ne 0 ]; then
      log "FAIL  $id ($run_name) rc=$rc — see $logf"
      touch "$SCHED_DIR/$id.failed"; exit $rc
    fi
    if [ "$kind" = "eval" ]; then
      local src="$EVAL_OUT_DIR/$run_name/eval_stats.csv"
      if [ -f "$src" ]; then
        cp "$src" "$(snap_path "$run_name")"
        log "SNAP  $id -> $(snap_path "$run_name")"
      else
        log "WARN  $id: expected $src not found; no snapshot written."
      fi
    fi
    log "DONE  $id ($run_name)"
    touch "$SCHED_DIR/$id.done"
  }
}

deps_met() {
  local deps="$1"
  [ "$deps" = "-" ] && return 0
  local d
  for d in ${deps//,/ }; do
    [ -f "$SCHED_DIR/$d.done" ] || return 1
  done
  return 0
}

running_count() {
  local c=0 pidf pid
  for pidf in "$SCHED_DIR"/*.pid; do
    [ -e "$pidf" ] || continue
    pid="$(cat "$pidf")"
    if kill -0 "$pid" 2>/dev/null; then c=$((c+1)); fi
  done
  echo "$c"
}

# --------------------------- scheduler -------------------------------------
schedule() {
  mkdir -p "$LOG_DIR" "$SCHED_DIR" "$EVAL_OUT_DIR" "$SNAP_DIR"
  log "Study output base: $EVAL_OUT_DIR"
  log "Snapshots:         $SNAP_DIR"
  log "runner=run.py (serial)  max_slots=$MAX_SLOTS concurrent experiments"

  while true; do
    # Abort if anything failed (do not start dependents on a broken pipeline).
    if ls "$SCHED_DIR"/*.failed >/dev/null 2>&1; then
      log "A job failed; stopping scheduler. Inspect $LOG_DIR, fix, and re-run to resume."
      wait; exit 1
    fi

    local all_done=1 spec id deps
    for spec in "${JOBS[@]}"; do
      id="$(field "$spec" 1)"
      [ -f "$SCHED_DIR/$id.done" ] && continue
      all_done=0
      # already launched and still tracked?
      [ -f "$SCHED_DIR/$id.pid" ] && kill -0 "$(cat "$SCHED_DIR/$id.pid")" 2>/dev/null && continue
      # launched, pid gone, but no .done and no .failed -> it died; treat as failure
      if [ -f "$SCHED_DIR/$id.pid" ]; then
        log "Job $id vanished without completing; marking failed."
        touch "$SCHED_DIR/$id.failed"; continue
      fi
      deps="$(field "$spec" 5)"
      deps_met "$deps" || continue
      if [ "$(running_count)" -lt "$MAX_SLOTS" ]; then
        run_job "$spec" &
        echo "$!" > "$SCHED_DIR/$id.pid"
        sleep 1  # let the child record its markers before we re-poll
      fi
    done

    [ "$all_done" -eq 1 ] && { log "All jobs complete."; break; }
    sleep "$POLL_SECS"
  done
  wait
}

# --------------------------- analysis --------------------------------------
analyze() {
  mkdir -p "$SNAP_DIR"
  log "Running paired p-value analysis..."
  local pair name la lb ra rb outf
  for pair in "${PAIRS[@]}"; do
    name="$(field "$pair" 1)"
    la="$(echo "$pair" | cut -d'|' -f2 | cut -d: -f1)"
    ra="$(echo "$pair" | cut -d'|' -f2 | cut -d: -f2)"
    lb="$(echo "$pair" | cut -d'|' -f3 | cut -d: -f1)"
    rb="$(echo "$pair" | cut -d'|' -f3 | cut -d: -f2)"
    local ca cb
    ca="$(snap_path "$ra")"; cb="$(snap_path "$rb")"
    outf="$SNAP_DIR/pvalue_$name.txt"
    if [ ! -f "$ca" ] || [ ! -f "$cb" ]; then
      log "SKIP analysis '$name' — missing snapshot(s): $ca / $cb"
      continue
    fi
    log "  $name: $la vs $lb -> $outf"
    "$PY" "$REPO/scripts/pvalue_test.py" "$ca" "$cb" \
        --label-a "$la" --label-b "$lb" \
        --metric monty_matching_steps --max-steps 500 \
        --csv "$SNAP_DIR/pvalue_${name}.csv" | tee "$outf"
    echo
  done
  log "Analysis written to $SNAP_DIR/pvalue_*.txt (+ .csv summaries)."
}

# --------------------------- entry -----------------------------------------
main() {
  require_env
  case "${1:-all}" in
    all)     schedule; analyze ;;
    run)     schedule ;;
    analyze) analyze ;;
    *) echo "usage: $0 [all|run|analyze]"; exit 2 ;;
  esac
}
main "$@"
