r"""jaxfdm_fold_spectra.py -- lambda_min of the adjoint operator at the limit point, JAX-FDM only
(Sec. 3.1, claims i and iii; Table 1 rows "lambda_min at V_f").

At the limit point of each mesh the smallest eigenvalue of the adjoint operator under
both closures is
    prescribed pressure   K_T  phi = lambda K phi          (Eq. 14)
    gas law               K~_T phi = lambda K phi          (Eq. 18)
This script takes the fold from jaxfdm_fold_ladder.py (results/jaxfdm/jaxfdm_fold_h<L>.npz),
re-solves the jax_fdm gas-law state there (K_gas = p*_h V*_h), and applies the tangents
matrix-free from jax_fdm's own load function:
    K_T v = K v - (dP/dx) v          (dP/dx by jax.jvp, P = jax_fdm nodal load)
with P frozen at p (prescribed) or following p = K_gas / V(x) (gas law). K^-1 is
jax_fdm's sparse solve. The symmetric-definite generalised problem is solved by
Lanczos (ARPACK eigsh, mode 2, M = K). No dense operator, so it runs at 1/48.
The symmetry of each tangent is also measured (v^T K_T w - w^T K_T v on random vectors).

Cross-check: the SciPy volume-control values at the same fold (cap_verify.py FOLD_ONLY=1:
lamP_fold, lamG_fold); they agree to 6 digits on every mesh.

RUN   TRI_LEVEL=6 python jaxfdm_fold_spectra.py            (after jaxfdm_fold_ladder.py)
OUT   results/jaxfdm/jaxfdm_fold_spectra_h<L><OUT_SUFFIX>.npz
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np                                           # noqa: E402
import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh        # noqa: E402

import jaxfdm_gaslaw.common as L                             # noqa: E402
from jaxfdm_gaslaw.common import (bp, structure, free_of, make_mass_model, THETA0, QDTYPE,  # noqa: E402
                                  DAMPED, V0, TMAX, N_DOF, UNIT_FACES, SIGN,
                                  _fixed_pressure_model)

RD = os.path.join(ROOT, "results", "jaxfdm")
LEVEL = L.TRI_LEVEL
SFX = os.environ.get("OUT_SUFFIX", "")
F = np.load(os.path.join(RD, f"jaxfdm_fold_h{LEVEL}{SFX}.npz"))
K_STAR = float(F["p_star"]) * float(F["V_star"])
OUT = os.path.join(RD, f"jaxfdm_fold_spectra_h{LEVEL}{SFX}.npz")
ETA = 1e-14
RES_TOL = 1e-12
Q = jnp.exp(THETA0).astype(QDTYPE)
N = N_DOF // 3


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


t0 = time.time()
print(f"jax_fdm fold spectra, h/R = 1/{LEVEL}, {N_DOF} dof; fold K* = {K_STAR:.8f}",
      flush=True)

# warm-started ladder up to the fold, then the fold state itself
xf = free_of(L.vertices)
gk = 0.25 * V0
while gk < K_STAR:
    xf, V, res = _solve(jnp.asarray(gk, dtype=QDTYPE), xf)
    gk *= 1.25
xf, V, res = _solve(jnp.asarray(K_STAR, dtype=QDTYPE), xf)
V, res = float(V), float(res)
if res >= RES_TOL:
    raise RuntimeError(f"fold state failed the equilibrium gate: residual {res:.2e}")
p = K_STAR / V
print(f"[state] V = {V:.10f} (ladder fit {float(F['V_star']):.10f}), p = {p:.10f}, "
      f"residual {res:.1e}, {time.time() - t0:.0f} s", flush=True)

mm = make_mass_model(K_STAR, ETA)
fm = _fixed_pressure_model(ETA)
K = mm.stiffness_matrix(Q, structure)
Rf = mm.residual_fixed_matrix(Q, bp.xyz_fixed, structure)
loads_f = bp.loads._replace(faces=UNIT_FACES * (SIGN * p))


def P_gas(x):
    return mm.load_xyz_matrix_from_r_fixed((K, Rf, bp.xyz_fixed, bp.loads), x, structure)


def P_presc(x):
    return fm.load_xyz_matrix_from_r_fixed((K, Rf, bp.xyz_fixed, loads_f), x, structure)


def tangent(Pfn):
    @jax.jit
    def kt(v):
        Vm = v.reshape(N, 3)
        return (K @ Vm - jax.jvp(Pfn, (xf,), (Vm,))[1]).ravel()
    return kt


K_mv = jax.jit(lambda v: (K @ v.reshape(N, 3)).ravel())
K_inv = jax.jit(lambda v: mm.linearsolve_fn(K, v.reshape(N, 3)).ravel())


def op(f):
    return LinearOperator((N_DOF, N_DOF), dtype=np.float64,
                          matvec=lambda v: np.asarray(f(jnp.asarray(v, dtype=QDTYPE))))


# symmetry of each tangent in the K-free sense: v^T K_T w - w^T K_T v
rng = np.random.default_rng(0)
v1, v2 = rng.standard_normal(N_DOF), rng.standard_normal(N_DOF)
out = {}
for name, Pfn in (("prescribed", P_presc), ("gas", P_gas)):
    kt = tangent(Pfn)
    a, b = np.asarray(kt(jnp.asarray(v1))), np.asarray(kt(jnp.asarray(v2)))
    asym = abs(v2 @ a - v1 @ b) / (np.linalg.norm(a) * np.linalg.norm(v2))
    t = time.time()
    lam, _ = eigsh(op(kt), k=3, M=op(K_mv), Minv=op(K_inv), which="SA",
                   tol=1e-13, ncv=40, maxiter=20000)
    lam = np.sort(lam)
    out[name] = lam
    print(f"[{name:10s}] lambda_1..3 = {lam[0]: .6e}  {lam[1]: .6f}  {lam[2]: .6f}   "
          f"asym {asym:.1e}   ({time.time() - t:.0f} s)", flush=True)

KS = L.scipy_fold(LEVEL)
lam_f_sp = float(np.ravel(KS["lamP_fold"])[0]) if KS is not None else np.nan
lam_m_sp = float(np.ravel(KS["lamG_fold"])[0]) if KS is not None else np.nan
print(f"\n{'':12}{'jax_fdm':>14}{'SciPy':>14}")
print(f"{'prescribed':12}{out['prescribed'][0]:14.3e}{lam_f_sp:14.3e}")
print(f"{'gas law':12}{out['gas'][0]:14.6f}{lam_m_sp:14.6f}")
np.savez(OUT, level=LEVEL, n_dof=N_DOF, K_star=K_STAR, V=V, p=p, residual=res,
         lam_presc=out["prescribed"], lam_gas=out["gas"],
         lam_f_scipy=lam_f_sp, lam_m_scipy=lam_m_sp,
         seconds=time.time() - t0)
print(f"[saved] {OUT}  ({time.time() - t0:.0f} s)")
