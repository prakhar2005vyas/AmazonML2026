#!/usr/bin/env bash
# Full rebuild of a `hard`-regime pipeline into a fresh work folder, for a plan-v2 stage that changes normalization,
# blocking or stage-1 features. Earlier work folders and submissions are never touched.
#
#   bash run_rebuild.sh <stage_name> <work_dir> [src_dir]
#
# Optional environment:
#   TAG=hard3 TTAG=v3     train / test tags passed to blocking.py and features.py (default: hard / none).
#                         stage2.TEST_SFX must map _$TAG to _$TTAG.
#   NORM_TRAIN=<dir>      hard-link translit.json and train_s*_norm.parquet from this folder instead of recomputing
#   NORM_TEST=<dir>       hard-link test_s*_norm.parquet from this folder instead of recomputing
#
# src_dir defaults to a frozen copy of ./src taken at start (so editing src mid-run is safe).
# Every step skips itself when its output already exists, so a crashed run is resumed by re-running.
set -euo pipefail
STAGE=$1
TAG=${TAG:-hard}
TTAG=${TTAG:-}
TS=${TTAG:+_$TTAG}   # "" or "_v3"
winpath() { if command -v cygpath >/dev/null; then cygpath -m "$1"; else echo "$1"; fi; }  # Windows Python needs C:/...
mkdir -p "$2"
export ER_WORK=$(winpath "$(cd "$2" && pwd)")
HERE=$(winpath "$(cd "$(dirname "$0")" && pwd)")
ROOT=$(winpath "$(cd "$HERE/.." && pwd)")
# config.py derives these from its own location, which is wrong for a frozen copy of src: pin them
export ER_DATA=${ER_DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}
export ER_OUTPUT=${ER_OUTPUT:-$ROOT/output}
BASE_WORK=${BASE_WORK:-$HOME/mlc26_work}
PY=${PY:-$HOME/.venvs/mlc26/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
SRC=${3:-$ER_WORK/src_frozen}
if [ -z "${3:-}" ] && [ ! -d "$SRC" ]; then cp -r "$HERE/src" "$SRC"; rm -rf "$SRC/__pycache__"; fi
cd "$SRC"
log() { echo "[$(date +%H:%M:%S)] $STAGE $*" | tee -a "$ER_WORK/log_rebuild.txt"; }
step() {  # step <done-marker> <log-name> <python args...>
  local marker=$1 name=$2; shift 2
  if [ -e "$ER_WORK/$marker" ]; then log "skip $name"; return; fi
  log "start $name"
  "$PY" "$@" > "$ER_WORK/log_$name.txt" 2>&1 || { log "FAILED $name (see log_$name.txt)"; exit 1; }
  log "done $name"
}
link() {  # link <src_dir> <file>: hard link (no copy) unless already present
  [ -e "$ER_WORK/$2" ] || ln "$1/$2" "$ER_WORK/$2" 2>/dev/null || cp "$1/$2" "$ER_WORK/$2"
}

# raw parquet conversions do not depend on any stage
for f in train_s1 train_s2 train_s3 test_s1 test_s2 test_s3 train_pairs; do link "$BASE_WORK" "$f.parquet"; done
if [ -n "${NORM_TRAIN:-}" ]; then
  link "$NORM_TRAIN" translit.json
  for s in 1 2 3; do link "$NORM_TRAIN" "train_s${s}_norm.parquet"; done
  log "train normalization linked from $NORM_TRAIN"
fi
if [ -n "${NORM_TEST:-}" ]; then
  for s in 1 2 3; do link "$NORM_TEST" "test_s${s}_norm.parquet"; done
  log "test normalization linked from $NORM_TEST"
fi

feats() {  # feats <marker-dir> <args...>: run features.py, then drop a .done marker
  local d=$1; shift
  echo "import runpy,sys,pathlib,os; sys.argv=['features.py'] + '$*'.split(); runpy.run_path('features.py', run_name='__main__'); pathlib.Path(os.environ['ER_WORK'],'$d','.done').touch()"
}

step translit.json                 translit    build_translit.py
[ -e "$ER_WORK/train_s3_norm.parquet" ] || step train_s3_norm.parquet normalize_train normalize.py train
step test_s3_norm.parquet          normalize_test normalize.py test
step "train_${TAG}_cand.parquet"   block_train blocking.py train "$TAG"
step "test${TS}_cand.parquet"      block_test  blocking.py test $TTAG
step "val_${TAG}_feats/.done"      feats_train -c "$(feats "val_${TAG}_feats" train "$TAG")"
step "test${TS}_feats/.done"       feats_test  -c "$(feats "test${TS}_feats" test $TTAG)"
step "decision_${TAG}.json"        model       model.py "$TAG"
step "decision2_${TAG}x2.json"     s2_val      stage2.py val "$TAG" --x2
step "log_err_${TAG}.done"         err         -c "import runpy,sys,pathlib,os; sys.argv=['analyze_errors.py','$TAG','0.75','val_${TAG}_oofx2.parquet']; runpy.run_path('analyze_errors.py', run_name='__main__'); pathlib.Path(os.environ['ER_WORK'],'log_err_${TAG}.done').touch()"
step "test_${TAG}_s2x2_scored.parquet" s2_test stage2.py test "$TAG" --x2 --out "$STAGE"
log "all steps done"
