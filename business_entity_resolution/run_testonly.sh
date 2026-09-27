#!/usr/bin/env bash
# Test-side-only rebuild for a plan-v2 stage whose changes cannot affect any training row (for example the France
# pack: France occurs only in test). It reuses the trained stage-1 and stage-2 models of an earlier work folder,
# and recomputes test normalization, blocking, features and scoring.
#
#   bash run_testonly.sh <stage_name> <model_work_dir> <new_work_dir> [src_dir]
#
# Every step skips itself when its output exists, so a crashed run is resumed by re-running.
set -euo pipefail
STAGE=$1
winpath() { if command -v cygpath >/dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
MODEL_WORK=$(winpath "$(cd "$2" && pwd)")
mkdir -p "$3"
export ER_WORK=$(winpath "$(cd "$3" && pwd)")
HERE=$(winpath "$(cd "$(dirname "$0")" && pwd)")
ROOT=$(winpath "$(cd "$HERE/.." && pwd)")
export ER_DATA=${ER_DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}
export ER_OUTPUT=${ER_OUTPUT:-$ROOT/output}
PY=${PY:-$HOME/.venvs/mlc26/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
SRC=${4:-$ER_WORK/src_frozen}
if [ -z "${4:-}" ] && [ ! -d "$SRC" ]; then cp -r "$HERE/src" "$SRC"; rm -rf "$SRC/__pycache__"; fi
cd "$SRC"
log() { echo "[$(date +%H:%M:%S)] $STAGE $*" | tee -a "$ER_WORK/log_rebuild.txt"; }
step() {
  local marker=$1 name=$2; shift 2
  if [ -e "$ER_WORK/$marker" ]; then log "skip $name"; return; fi
  log "start $name"
  "$PY" "$@" > "$ER_WORK/log_$name.txt" 2>&1 || { log "FAILED $name (see log_$name.txt)"; exit 1; }
  log "done $name"
}

# inputs and trained models from the model work folder (hard links, so nothing is copied)
for f in test_s1.parquet test_s2.parquet test_s3.parquet train_pairs.parquet translit.json \
         lgb_hard.txt lgb2_hardx2.txt decision2_hardx2.json; do
  [ -e "$ER_WORK/$f" ] || ln "$MODEL_WORK/$f" "$ER_WORK/$f" 2>/dev/null || cp "$MODEL_WORK/$f" "$ER_WORK/$f"
done

step test_s3_norm.parquet  normalize   normalize.py test
step test_cand.parquet     block_test  blocking.py test
step test_feats/.done      feats_test  -c "import runpy,sys,pathlib,os; sys.argv=['features.py','test']; runpy.run_path('features.py', run_name='__main__'); pathlib.Path(os.environ['ER_WORK'],'test_feats','.done').touch()"
step test_hard_s2x2_scored.parquet s2_test stage2.py test hard --x2 --out "$STAGE"
log "all steps done"
