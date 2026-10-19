#!/usr/bin/env bash
# Reproduce the JAX-FDM side of Sec. 3.1 (gas-law closure inside DFDM) at the meshes in LEVELS.
#
#   PYTHON=/path/to/python LEVELS="6 12" ./run_jaxfdm.sh
#
# Needs the patched jax_fdm (python jaxfdm_gaslaw/patch/apply_patch.py --check) and the SciPy
# fold files of run_sec31.sh (or the shipped ones in results/sec31/paper).
# Defaults: LEVELS="6 12" (about 30 min on a laptop). The dense adjoint reference and the FD check
# through the fold are run up to 1/12 only (1/24 needs ~16 GB). 1/48 needs JAXFDM_COMPACT=1.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
LEVELS=${LEVELS:-"6 12"}
LOG=results/logs
mkdir -p "$LOG"

run() {
  local name=$1; shift
  echo "[$(date +%T)] $name"
  "$@" > "$LOG/$name.log" 2>&1
}

for L in $LEVELS; do
  SFX=""
  COMPACT=0
  if [ "$L" -ge 48 ]; then SFX="_compact"; COMPACT=1; fi
  run jax_fold_$L    env TRI_LEVEL=$L JAXFDM_COMPACT=$COMPACT OUT_SUFFIX=$SFX "$PY" sec31/jaxfdm_fold_ladder.py
  run jax_spectra_$L env TRI_LEVEL=$L JAXFDM_COMPACT=$COMPACT OUT_SUFFIX=$SFX "$PY" sec31/jaxfdm_fold_spectra.py
  if [ "$L" -le 12 ]; then
    run jax_smref_u$L env TRI_LEVEL=$L "$PY" sec31/jaxfdm_adjoint_reference.py
  fi
  run jax_pole_$L    env TRI_LEVEL=$L JAXFDM_COMPACT=$COMPACT OUT_SUFFIX=$SFX "$PY" sec31/jaxfdm_pole_exponent.py
  if [ "$L" -le 12 ]; then
    run jax_fd_u$L env TRI_LEVEL=$L "$PY" sec31/jaxfdm_fd_through_fold.py
  fi
  if [ "$L" -ge 12 ] && [ "$L" -le 12 ]; then
    run jax_smref_z$L env TRI_LEVEL=$L ZONED=1 "$PY" sec31/jaxfdm_adjoint_reference.py
    run jax_fd_z$L    env TRI_LEVEL=$L ZONED=1 "$PY" sec31/jaxfdm_fd_through_fold.py
  fi
done
echo "[$(date +%T)] done; logs in $LOG"
