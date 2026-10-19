#!/usr/bin/env bash
# The design runs of claim (vi) / Fig. 3(f): prescribed pressure against the gas law towards a
# target just past its own limit point, three starts below the limit point per cap.
#
#   PYTHON=/path/to/python LEVEL=6 CASES="uniform" ./run_capopt.sh        (~20 min on a laptop)
#   PYTHON=/path/to/python LEVEL=24 CASES="uniform zoned" ./run_capopt.sh  (the paper's runs: hours each;
#                                                                           the results are in
#                                                                           results/capopt_fold/paper)
# Needs the SciPy fold files (run_sec31.sh, or the shipped ones). Output files are never overwritten:
# set OUT_DIR to rerun a case.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
LEVEL=${LEVEL:-6}
CASES=${CASES:-"uniform"}
STARTS=${STARTS:-"0.4 0.7 0.9895"}
LOG=results/logs
mkdir -p "$LOG"

for C in $CASES; do
  for S in $STARTS; do
    echo "[$(date +%T)] capopt $C h$LEVEL start $S"
    env TRI_LEVEL=$LEVEL CASE=$C START=$S "$PY" sec31/cap_optimise_target.py > "$LOG/capopt_${C}_h${LEVEL}_s${S}.log" 2>&1
  done
done
echo "[$(date +%T)] done; logs in $LOG"
