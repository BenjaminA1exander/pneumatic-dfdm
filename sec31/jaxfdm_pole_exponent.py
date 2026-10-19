r"""jaxfdm_pole_exponent.py -- gradient-norm exponent n near the limit point, JAX-FDM only
(Sec. 3.1, claim ii: the pole of the prescribed-pressure gradient).

||dL/dtheta|| ~ (1 - V/V*_h)^n fitted over 1 - V/V*_h < 0.02, for prescribed pressure
(simple pole, n = -1) and for the gas law (bounded, n = 0). The norms come from the
jax_fdm adjoint itself (jax.grad through the implicit fixed point), with the GMRES
backward rule applying K^-1 by jax_fdm's sparse solve ("gmres_sparse"), so it runs
at h/R = 1/48 as well.

Stations: K_gas = s K*_h with s = 1 - geomspace(0.15, 1e-4, 12), K*_h = p*_h V*_h from
jaxfdm_fold_h<L><SFX>.npz. Gas-law states by warm-started continuation; the
prescribed-pressure gradient at the same state with p = K_gas / V (the state is a
fixed point of the prescribed problem too; its forward drift is recorded).

Check: where the dense Sherman-Morrison reference exists for this mesh
(jaxfdm_adjoint_reference.py -> limitpoint_smref_tri<L>*.npz) the norms are compared
with it at the same stations.

RUN   TRI_LEVEL=6 python jaxfdm_pole_exponent.py           (after jaxfdm_fold_ladder.py)
      TRI_LEVEL=48 JAXFDM_COMPACT=1 OUT_SUFFIX=_compact python jaxfdm_pole_exponent.py
OUT   results/jaxfdm/jaxfdm_pole_h<L><OUT_SUFFIX>.npz
"""
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np                                           # noqa: E402
import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

import jaxfdm_gaslaw.common as L                             # noqa: E402
from jaxfdm_gaslaw.common import (bp, structure, free_of, make_mass_model, mass_xyz,  # noqa: E402
                                  loss_from_xyz, use_adjoint, THETA0, QDTYPE, DAMPED, V0,
                                  TMAX, N_DOF, UNIT_FACES, SIGN, _fixed_pressure_model)

RD = os.path.join(ROOT, "results", "jaxfdm")
LEVEL = L.TRI_LEVEL
SFX = os.environ.get("OUT_SUFFIX", "")
F = np.load(os.path.join(RD, f"jaxfdm_fold_h{LEVEL}{SFX}.npz"))
K_STAR, V_STAR = float(F["p_star"]) * float(F["V_star"]), float(F["V_star"])
OUT = os.path.join(RD, f"jaxfdm_pole_h{LEVEL}{SFX}.npz")
ETA = 1e-14
RES_TOL = 1e-12
DRIFT_TOL = 1e-10
S = 1.0 - np.geomspace(0.15, 1e-4, 12)
Q = jnp.exp(THETA0).astype(QDTYPE)

use_adjoint("gmres_sparse")


@jax.jit
def _solve(gk, xf_init):
    mm = make_mass_model(gk, ETA)
    xf = mm.equilibrium_iterative_xyz(Q, bp.xyz_fixed, bp.loads, structure,
                                      xyz_free_init=xf_init, tmax=TMAX, eta=ETA,
                                      solver=DAMPED, implicit_diff=True)
    K = mm.stiffness_matrix(Q, structure)
    Rf = mm.residual_fixed_matrix(Q, bp.xyz_fixed, structure)
    P = mm.load_xyz_matrix_from_r_fixed((K, Rf, bp.xyz_fixed, bp.loads), xf, structure)
    res = jnp.mean(jnp.linalg.norm(mm.linearsolve_fn(K, P) - xf, axis=1))
    V = jnp.abs(mm.enclosed_volume(mm.nodes_positions(xf, bp.xyz_fixed, structure)))
    return xf, V, res


@jax.jit
def _grad_gas(theta, gk, xf):
    """dL/dtheta under the gas law: jax.grad through the implicit fixed point (Eq. 14 with K~_T)."""
    mm = make_mass_model(gk, ETA)
    return jax.grad(lambda th: loss_from_xyz(mass_xyz(mm, th, xf, ETA)))(theta)


def _presc_xf(theta, p, xf):
    fm = _fixed_pressure_model(ETA)
    loads = bp.loads._replace(faces=UNIT_FACES * (SIGN * p))
    return fm, fm.equilibrium_iterative_xyz(
        jnp.exp(theta).astype(QDTYPE), bp.xyz_fixed, loads, structure,
        xyz_free_init=xf, tmax=TMAX, eta=ETA, solver=DAMPED, implicit_diff=True)


@jax.jit
def _grad_presc(theta, p, xf):
    """dL/dtheta under prescribed pressure, same state (Eq. 14 with K_T)."""
    def loss(th):
        fm, x = _presc_xf(th, p, xf)
        return loss_from_xyz(fm.nodes_positions(x, bp.xyz_fixed, structure))
    return jax.grad(loss)(theta)


@jax.jit
def _drift(p, xf):
    return jnp.max(jnp.abs(_presc_xf(THETA0, p, xf)[1] - xf))


t0 = time.time()
print(f"jax_fdm pole exponent, h/R = 1/{LEVEL}, {N_DOF} dof, K* = {K_STAR:.8f}", flush=True)
xf = free_of(L.vertices)
gk = 0.25 * V0
while gk < S[0] * K_STAR:
    xf = _solve(jnp.asarray(gk, dtype=QDTYPE), xf)[0]
    gk *= 1.25

rows = []
for s in S:
    gk = jnp.asarray(s * K_STAR, dtype=QDTYPE)
    xf, V, res = _solve(gk, xf)
    V, res = float(V), float(res)
    if res >= RES_TOL:
        raise RuntimeError(f"s={s}: residual {res:.1e}")
    p = jnp.asarray(float(gk) / V, dtype=QDTYPE)
    t = time.time()
    g_m = np.asarray(_grad_gas(THETA0, gk, xf))
    g_f = np.asarray(_grad_presc(THETA0, p, xf))
    drift = float(_drift(p, xf))
    d = 1.0 - V / V_STAR
    rows.append((s, V / V_STAR, d, np.linalg.norm(g_m), np.linalg.norm(g_f), drift, res))
    print(f"  s={s:.6f}  1-V/V*={d:.3e}  |g_gas|={rows[-1][3]:.6e}  |g_presc|={rows[-1][4]:.6e}"
          f"  drift={drift:.1e}  ({time.time() - t:.0f} s)", flush=True)

R = np.array(rows)
if (R[:, 5] > DRIFT_TOL).any():
    print(f"[warn] prescribed forward solve left the state at {int((R[:, 5] > DRIFT_TOL).sum())} stations")
k = (R[:, 2] > 0) & (R[:, 2] < 0.02)
n_f = np.polyfit(np.log10(R[k, 2]), np.log10(R[k, 4]), 1)[0]
n_m = np.polyfit(np.log10(R[k, 2]), np.log10(R[k, 3]), 1)[0]
print(f"\n[pole] h/R = 1/{LEVEL}: n_presc = {n_f:.3f}, n_gas = {n_m:+.4f} "
      f"({int(k.sum())} stations, {time.time() - t0:.0f} s)")

sm = sorted(glob.glob(os.path.join(RD, f"limitpoint_smref_tri{LEVEL}*.npz"))
            + glob.glob(os.path.join(RD, "paper", f"limitpoint_smref_tri{LEVEL}*.npz")))
if sm:
    D = np.load(sm[-1])
    for name, col, j in (("gas", "ng_m", 3), ("presc", "ng_f", 4)):
        dev = [abs(R[i, j] - D[col][np.argmin(np.abs(D["s"] * float(D["k_crit"]) - s * K_STAR))])
               / R[i, j] for i, s in enumerate(R[:, 0])]
        print(f"[check] |g_{name}| vs dense reference ({os.path.basename(sm[-1])}) at the same K_gas: "
              f"max rel diff {max(dev):.1e}")

np.savez(OUT, level=LEVEL, n_dof=N_DOF, s=R[:, 0], Vfrac=R[:, 1], d=R[:, 2],
         ng_gas=R[:, 3], ng_presc=R[:, 4], drift=R[:, 5], residual=R[:, 6],
         n_presc=n_f, n_gas=n_m, seconds=time.time() - t0)
print(f"[saved] {OUT}")
