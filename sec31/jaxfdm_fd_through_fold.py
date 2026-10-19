r"""jaxfdm_fd_through_fold.py -- finite-difference check of the JAX-FDM gas-law gradient
THROUGH the limit point (Sec. 3.1: "central finite differences of the nonlinear forward
solve, taken along the gradient, agree with the JAX-FDM gradient to 1e-11 / 7e-10").

DFDM with K~_T in place of K_T returns the correct gradient through the limit point. The
dense reference of jaxfdm_adjoint_reference.py only checks that JAX-FDM solves its own
adjoint system correctly; it shares the model and the adjoint equation. This script checks
the gradient against the nonlinear forward solve alone:

    D(t) = [L(theta + t d) - L(theta - t d)] / 2t,   d = g / |g|,
    g = JAX-FDM gas-law gradient (jax.grad, implicit adjoint)

with L(theta) = loss_from_xyz(mass_xyz(model, theta, ...)): the gas-law equilibrium
re-solved by the JAX-FDM forward iteration at each perturbed theta, never differentiated.
Under the gas law the tangent is positive definite through the fold, so the perturbed
solves are as well conditioned there as anywhere else. Along d the exact directional
derivative is |g|; reported: |D(t) - |g|| / |g| for t = 1e-2 .. 1e-5 and the Richardson
combination [4 D(t/2) - D(t)] / 3 (removes the O(t^2) truncation term). Every perturbed
solve must pass the equilibrium gate (fixed-point residual < 1e-12).

Stations: gas content K = s K_f, s = 0.5 ... 2.5, with K_f = p_f V_f the limit point of the
same discrete cap from the SciPy code (cap_verify.py FOLD_ONLY=1); s = 1 is the fold.

RUN   TRI_LEVEL=12 python jaxfdm_fd_through_fold.py                              (uniform cap, ~5 min)
      TRI_LEVEL=12 ZONED=1 python jaxfdm_fd_through_fold.py                      (zoned cap)
      h/R = 1/24 needs about 16 GB of memory.
ENV   ZONED (0), GAMMA (0.25), BETA (10), RB (0.5), BELT_W (0.125), BLEND (1): the zoned cap
OUT   results/jaxfdm/fd_through_fold_<uniform|zoned>_tri<L>_<time>.npz + stdout table
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

import jaxfdm_gaslaw.common as L                             # noqa: E402

ZONED = os.environ.get("ZONED", "0") == "1"
GAMMA = float(os.environ.get("GAMMA", "0.25"))
BETA = float(os.environ.get("BETA", "10"))
RB = float(os.environ.get("RB", "0.5"))
BW = float(os.environ.get("BELT_W", "0.125"))
TAG = "zoned" if ZONED else "uniform"
RD = os.path.join(ROOT, "results", "jaxfdm")
os.makedirs(RD, exist_ok=True)
OUT = os.path.join(RD, f"fd_through_fold_{TAG}_tri{L.TRI_LEVEL}_{time.strftime('%Y%m%d-%H%M%S')}.npz")
ETA = 1e-14

if ZONED:
    from fdm_limit_point import zones
    th0, nz = L.zoned_theta0(GAMMA, BETA, RB, BW)
    F = L.scipy_fold(L.TRI_LEVEL, zoned=True, zone_tag=f"_w{BW:g}" + ("_blend" if zones.BLEND else ""))
else:
    th0, nz = L.THETA0, None
    F = L.scipy_fold(L.TRI_LEVEL)
if F is None:
    raise SystemExit("no SciPy fold file for this cap and mesh: run cap_verify.py with FOLD_ONLY=1 first")
V_CRIT, P_CRIT = float(F["Vf"]), float(F["pf"])
K_CRIT = P_CRIT * V_CRIT

# gas content K = s K_crit; s = 1 is the limit point. Ascending, so each state warm-starts the next.
S = np.array([0.5, 0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1, 1.5, 2.5])
TS = np.array([1e-2, 1e-3, 1e-4, 1e-5])


def resid(mm, th, xf):
    """Fixed-point residual of a free-node state: mean |x - K^-1 P(x)|, the equilibrium gate."""
    q = jnp.exp(th).astype(L.QDTYPE)
    K = mm.stiffness_matrix(q, L.structure)
    Rf = mm.residual_fixed_matrix(q, L.bp.xyz_fixed, L.structure)
    P = mm.load_xyz_matrix_from_r_fixed((K, Rf, L.bp.xyz_fixed, L.bp.loads), xf, L.structure)
    return float(jnp.mean(jnp.linalg.norm(mm.linearsolve_fn(K, P) - xf, axis=1)))


if __name__ == "__main__":
    t0 = time.time()
    print(f"fd_through_fold [{TAG}, tri{L.TRI_LEVEL}, {L.N_DOF} dof, {L.N_EDGES} edges"
          f"{'' if nz is None else ', edges soft/belt/outer ' + str(list(nz))}]  K_f {K_CRIT:.6f}  "
          f"V_f {V_CRIT:.6f}  (SciPy fold)", flush=True)
    rows, seed = [], L.free_of(L.vertices)
    for s in S:
        mm = L.make_mass_model(float(s) * K_CRIT, ETA)
        xyz = L.mass_xyz(mm, th0, seed, ETA)
        xf = L.free_of(xyz)
        V = float(L.enclosed_volume_np(np.asarray(xyz), L.faces_tri))
        r0 = resid(mm, th0, xf)

        def loss_th(th, x0):
            return L.loss_from_xyz(L.mass_xyz(mm, th, x0, ETA))

        g = np.asarray(jax.jit(jax.grad(loss_th))(th0, xf))          # JAX-FDM gas-law gradient (implicit adjoint)
        gn = float(np.linalg.norm(g))
        d = jnp.asarray(g / gn, dtype=L.QDTYPE)
        Lf = jax.jit(loss_th)
        D, rmax = [], 0.0
        for t in TS:
            lp = float(Lf(th0 + t * d, xf))
            lm = float(Lf(th0 - t * d, xf))
            D.append((lp - lm) / (2 * t))
            for sg in (1, -1):                                       # every perturbed solve must be an equilibrium
                xs = L.free_of(L.mass_xyz(mm, th0 + sg * t * d, xf, ETA))
                rmax = max(rmax, resid(mm, th0 + sg * t * d, xs))
        D = np.array(D)
        err = np.abs(D - gn) / gn
        rich = (4 * D[1:] - D[:-1]) / 3
        err_r = np.abs(rich - gn) / gn
        row = dict(s=float(s), Vfrac=V / V_CRIT, resid0=r0, resid_pert=rmax, gnorm=gn,
                   **{f"err_t{k}": e for k, e in enumerate(err)}, **{f"errR_{k}": e for k, e in enumerate(err_r)})
        rows.append(row)
        print(f"s {s:6.3f}  V/V_f {V / V_CRIT:7.4f}  |g| {gn:.3e}  resid {r0:.1e}/{rmax:.1e}  "
              f"FD err (t=1e-2..1e-5) " + " ".join(f"{e:.1e}" for e in err) +
              "  Richardson " + " ".join(f"{e:.1e}" for e in err_r) + f"  ({time.time() - t0:.0f} s)", flush=True)
        np.savez(OUT, ts=TS, **{k: np.array([q[k] for q in rows]) for k in rows[0]})
        seed = xf
    best = np.array([min([q[k] for k in q if k.startswith("err")]) for q in rows])
    print(f"[saved] {OUT}\nbest FD agreement per state: " + " ".join(f"{b:.1e}" for b in best) +
          f"\nworst of the best: {best.max():.1e}  ({time.time() - t0:.0f} s)")
