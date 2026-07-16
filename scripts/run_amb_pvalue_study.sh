#!/usr/bin/env bash
#
# run_amb_pvalue_study.sh — orchestrate the ambient LTP-vs-no-LTP p-value study.
#
# Runs 3 trainings + 7 evals with at most 2 experiments in flight at once,
# packed to finish fastest (the >10h 77obj training starts immediately in one slot
# while the short 11retex trainings + all 11retex evals stream through the other;
# the two 77obj evals fill both slots once the 77obj model is ready). Then runs the
# paired p-value analysis for each LTP-vs-no-LTP pair.
#
# Each experiment is launched with the SERIAL run.py (not run_parallel.py) -- the
# only concurrency is the 2 independent run.py processes this scheduler runs side by
# side, which avoids run_parallel's multiprocessing Pool (flaky on Mac). Set
# MAX_SLOTS=1 if you want strictly one experiment at a time.
#
# Idempotent / resumable:
#   * a training is SKIPPED if its model.pt already exists (reuses trained models),
#   * an eval is SKIPPED if its snapshot CSV already exists.
# So you can re-run this script to resume after an interruption.
#
# Prerequisites:
#   conda activate tbp.monty
#   export MONTY_MODELS / MONTY_LOGS / MONTY_DATA   (as for your normal runs)
#
# Usage:
#   scripts/run_amb_pvalue_study.sh            # run everything (train+eval+analyze)
#   MAX_SLOTS=1 scripts/run_amb_pvalue_study.sh   # strictly one experiment at a time
#   scripts/run_amb_pvalue_study.sh analyze    # only (re)run the p-value analysis
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
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${MONTY_LOGS:-$HOME/tbp/results/monty}/projects/monty_runs_pval_amb}"
SNAP_DIR="${SNAP_DIR:-$EVAL_OUT_DIR/_snapshots}"
LOG_DIR="$SNAP_DIR/logs"
SCHED_DIR="$SNAP_DIR/.sched"

MODELS_ROOT="${MONTY_MODELS:-$HOME/tbp/results/monty/pretrained_models}/my_trained_models"

# --------------------------- job table -------------------------------------
# id | kind(train|eval) | hydra experiment config | run_name | deps(comma or -)
# run_name for train == model dir name; for eval == output subdir + snapshot key.
# Order = launch priority (long-pole 77obj training first).
JOBS=(
  "t1|train|pval_amb_train_77obj_ltp|amb_surf_pre_training_77obj_ltp|-"
  "t2|train|pval_amb_train_11retex_multiscale|amb_surf_pre_training_11retextured_obj_ltp_multiscale|-"
  "t3|train|pval_amb_train_11retex_uniform_rgb|amb_surf_pre_training_11retextured_obj_ltp_uniform_rgb|-"
  "e3|eval|pval_amb_eval_11retex_ltp|pval_amb_11retex_surf_agent_ltp|-"
  "e4|eval|pval_amb_eval_11retex_noltp|pval_amb_11retex_surf_agent_noltp|-"
  "e5|eval|pval_amb_eval_11retex_multiscale|pval_amb_11retex_surf_agent_ltp_multiscale|t2"
  "e6|eval|pval_amb_eval_11retex_uniform|pval_amb_11retex_surf_agent_ltp_uniform|t3"
  "e7|eval|pval_amb_eval_11retex_rgb|pval_amb_11retex_surf_agent_ltp_rgb|t3"
  "e1|eval|pval_amb_eval_77obj_ltp|pval_amb_77obj_surf_agent_ltp|t1"
  "e2|eval|pval_amb_eval_77obj_noltp|pval_amb_77obj_surf_agent_noltp|t1"
  # HSV+LTP arms (added later): HSV stays weighted AND LTP is added, so pairing each
  # against the HSV-only baseline tests whether LTP ADDS signal on top of color.
  # Reuse the same trained models/sensors as their LTP-only counterparts (no retraining).
  "e3b|eval|pval_amb_eval_11retex_hsvltp|pval_amb_11retex_surf_agent_hsvltp|-"
  "e5b|eval|pval_amb_eval_11retex_multiscale_hsvltp|pval_amb_11retex_surf_agent_ltp_multiscale_hsvltp|t2"
  "e6b|eval|pval_amb_eval_11retex_uniform_hsvltp|pval_amb_11retex_surf_agent_ltp_uniform_hsvltp|t3"
  "e7b|eval|pval_amb_eval_11retex_rgb_hsvltp|pval_amb_11retex_surf_agent_ltp_rgb_hsvltp|t3"
  "e1b|eval|pval_amb_eval_77obj_hsvltp|pval_amb_77obj_surf_agent_hsvltp|t1"
)

# LTP-vs-no-LTP pairs to test at the end: "label_ltp:run_ltp|label_base:run_base"
PAIRS=(
  "77obj|LTP:pval_amb_77obj_surf_agent_ltp|noLTP:pval_amb_77obj_surf_agent_noltp"
  "11retex|LTP:pval_amb_11retex_surf_agent_ltp|noLTP:pval_amb_11retex_surf_agent_noltp"
  "11retex_multiscale|multiscale:pval_amb_11retex_surf_agent_ltp_multiscale|noLTP:pval_amb_11retex_surf_agent_noltp"
  "11retex_uniform|uniform:pval_amb_11retex_surf_agent_ltp_uniform|noLTP:pval_amb_11retex_surf_agent_noltp"
  "11retex_rgb|rgb:pval_amb_11retex_surf_agent_ltp_rgb|noLTP:pval_amb_11retex_surf_agent_noltp"
  # HSV+LTP vs HSV-only baseline: the primary "does LTP add to color?" test.
  "77obj_hsvltp|HSV+LTP:pval_amb_77obj_surf_agent_hsvltp|HSV:pval_amb_77obj_surf_agent_noltp"
  "11retex_hsvltp|HSV+LTP:pval_amb_11retex_surf_agent_hsvltp|HSV:pval_amb_11retex_surf_agent_noltp"
  "11retex_multiscale_hsvltp|HSV+multiscale:pval_amb_11retex_surf_agent_ltp_multiscale_hsvltp|HSV:pval_amb_11retex_surf_agent_noltp"
  "11retex_uniform_hsvltp|HSV+uniform:pval_amb_11retex_surf_agent_ltp_uniform_hsvltp|HSV:pval_amb_11retex_surf_agent_noltp"
  "11retex_rgb_hsvltp|HSV+rgb:pval_amb_11retex_surf_agent_ltp_rgb_hsvltp|HSV:pval_amb_11retex_surf_agent_noltp"
  # HSV+LTP vs LTP-only: isolates what color adds once you already have texture.
  "11retex_hsvltp_vs_ltp|HSV+LTP:pval_amb_11retex_surf_agent_hsvltp|LTP:pval_amb_11retex_surf_agent_ltp"
  "77obj_hsvltp_vs_ltp|HSV+LTP:pval_amb_77obj_surf_agent_hsvltp|LTP:pval_amb_77obj_surf_agent_ltp"
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
