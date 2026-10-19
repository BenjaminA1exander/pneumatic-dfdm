#!/usr/bin/env bash
# Reproduce the SciPy side of Sec. 3.1 (Table 1, Fig. 2, Fig. 3) at the meshes in LEVELS.
#
#   PYTHON=/path/to/python LEVELS="6 12 24" ./run_sec31.sh
#
# Defaults: LEVELS="6 12 24" (about 25 min on a laptop; 1/48 adds ~1.5 h and needs ~2 GB).
# The zoned cap starts at 1/12 (at 1/6 the belt is narrower than one element).
# Every run writes a new time-stamped file into results/sec31/; logs go to results/logs/.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
LEVELS=${LEVELS:-"6 12 24"}
LOG=results/logs
mkdir -p "$LOG"

run() {
  local name=$1; shift
  echo "[$(date +%T)] $name"
  "$@" > "$LOG/$name.log" 2>&1
}

for L in $LEVELS; do
  run fold_u$L   env CAP=uniform TRI_LEVEL=$L FOLD_ONLY=1 "$PY" sec31/cap_verify.py
  run verify_u$L env CAP=uniform TRI_LEVEL=$L             "$PY" sec31/cap_verify.py
  run probe_u$L  env CAP=uniform TRI_LEVEL=$L             "$PY" sec31/cap_closure_probe.py
  if [ "$L" -ge 12 ]; then
    run fold_z$L   env CAP=zoned TRI_LEVEL=$L FOLD_ONLY=1 "$PY" sec31/cap_verify.py
    run verify_z$L env CAP=zoned TRI_LEVEL=$L             "$PY" sec31/cap_verify.py
    run probe_z$L  env CAP=zoned TRI_LEVEL=$L             "$PY" sec31/cap_closure_probe.py
    run bubble_z$L env CAP=zoned TRI_LEVEL=$L             "$PY" sec31/cap_bubble_check.py
  fi
done

# the volume-controlled paths of Fig. 3(b,c) at h/R = 1/24, the zoned one to 7 V_f (S at 6.7 V_f)
if echo " $LEVELS " | grep -q " 24 "; then
  run pin_u24 env CAP=uniform TRI_LEVEL=24            "$PY" sec31/cap_model.py
  run pin_z24 env CAP=zoned   TRI_LEVEL=24 V_END=1.56 "$PY" sec31/cap_model.py
fi

run table1 "$PY" sec31/table1.py
run fig2   "$PY" sec31/fig2_method.py
if echo " $LEVELS " | grep -q " 24 "; then
  run fig3 "$PY" sec31/fig3_verification.py
fi
echo "[$(date +%T)] done; see $LOG/table1.log"
