r"""jaxfdm_adjoint_reference.py -- both DFDM adjoints against a dense, non-finite-difference
reference, through and past the limit point (Sec. 3.1: "the gradient was compared with a
dense direct solve of the same adjoint system, obtained from the gas-law system by the
Sherman-Morrison formula; it agrees to 1e-12 away from the limit point and 1e-9 next to it").

Near the fold a finite-difference reference degrades together with the prescribed-pressure
gradient, and it cannot follow the prescribed problem to or past the fold at all. On the
coarse meshes the adjoint systems can be solved DENSELY instead, which gives a reference
whose error is set by conditioning and machine precision only:

    gas law              w_m = A_m^-T v                     (A_m well conditioned)
    prescribed pressure  w_f = w_m - y (a^T w_m) / delta,   y = A_m^-T b
                         (Sherman-Morrison on A_f = A_m + a b^T: A_f - A_m is exactly
                          rank one for one chamber, Eq. 18; only solves against the
                          well-conditioned A_m) and, for cross-checking, w_f = A_f^-T v
                          by dense LU

and g = w^T dF/dtheta for each. The jax_fdm gradients are computed with the GMRES adjoint
rule at rtol 1e-12 and the same forward tolerance.

The prescribed-pressure gradient is evaluated at the SAME converged states as the gas-law
one (p frozen at K_gas/V), which are fixed points of the prescribed problem on both sides
of the fold. Past the fold they are unstable fixed points: the prescribed forward iteration
is started on them and its drift is recorded; a station whose forward solve leaves the
state is excluded rather than reported.

Stations: K_gas = s K_f with the fold K_f = p_f V_f from the SciPy code of the same cap
(cap_verify.py FOLD_ONLY=1): 8 far below, 12 approaching to 1 - 1e-4, the fold itself,
8 just past it, 4 far past it.

RUN   TRI_LEVEL=6 python jaxfdm_adjoint_reference.py                 (uniform cap; 1/6 ~1 min, 1/12 ~10 min)
      TRI_LEVEL=12 ZONED=1 python jaxfdm_adjoint_reference.py        (zoned cap)
      Dense: memory grows as (3 n_free)^2; 1/24 (5979 dof) needs ~16 GB.
ENV   ZONED (0), GAMMA, BETA, RB, BELT_W, BLEND (zoned cap), N_PROBE (40), SMREF_SUFFIX
OUT   results/jaxfdm/limitpoint_smref_tri<L>.npz or
      results/jaxfdm/limitpoint_smref_zoned_g<..>_b<..>_w<..>[_blend]_tri<L>.npz
      per station s: V/V_f, p/p_f, sigma_min of both operators, delta, the relative errors of
      each jax_fdm adjoint against its dense reference, and S-M against dense LU.
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
from jaxfdm_gaslaw.common import (                           # noqa: E402
    bp, structure, faces_tri, enclosed_volume_np, loss_from_xyz, free_of,
    make_mass_model, mass_xyz, use_adjoint, SIGN, UNIT_FACES, QDTYPE,
    N_EDGES, N_DOF, DAMPED, _fixed_pressure_model,
)

RD = os.path.join(ROOT, "results", "jaxfdm")
os.makedirs(RD, exist_ok=True)
ZONED = os.environ.get("ZONED", "0") == "1"
if ZONED:
    from fdm_limit_point import zones
    GAMMA = float(os.environ.get("GAMMA", "0.25"))
    BETA = float(os.environ.get("BETA", "10"))
    RB = float(os.environ.get("RB", "0.5"))
    BW = float(os.environ.get("BELT_W", "0.125"))
    _BT = "_blend" if zones.BLEND else ""
    THETA0, _nz = L.zoned_theta0(GAMMA, BETA, RB, BW)
    F = L.scipy_fold(L.TRI_LEVEL, zoned=True, zone_tag=f"_w{BW:g}{_BT}")
    OUT = L.out_path(f"limitpoint_smref_zoned_g{GAMMA:g}_b{BETA:g}_w{BW:g}{_BT}", RD)
    CASE = f"zoned GAMMA {GAMMA} BETA {BETA} RB {RB} W {BW}, edges soft/belt/outer {list(_nz)}"
else:
    THETA0 = L.THETA0
    F = L.scipy_fold(L.TRI_LEVEL)
    OUT = L.out_path("limitpoint_smref", RD).replace(".npz", os.environ.get("SMREF_SUFFIX", "") + ".npz")
    CASE = "uniform"
if F is None:
    raise SystemExit("no SciPy fold file for this cap and mesh: run cap_verify.py with FOLD_ONLY=1 first")
V_CRIT, P_CRIT = float(F["Vf"]), float(F["pf"])
K_CRIT = P_CRIT * V_CRIT
DENSE_SVD = N_DOF <= 2000      # sigma_min by dense SVD only where affordable

ETA = 1e-14
TMAX = 6000
RES_TOL = 1e-12             # equilibrium gate
DRIFT_TOL = 1e-10           # prescribed forward solve must stay on the state (x R)

# a fixed random subset of edges, on which the errors are also reported
N_PROBE = int(os.environ.get("N_PROBE", 40))
PROBE = np.sort(np.random.default_rng(0).choice(
    N_EDGES, size=min(N_PROBE, N_EDGES), replace=False))

use_adjoint("gmres")

S = np.unique(np.concatenate([
    np.geomspace(0.05, 0.80, 8),
    1.0 - np.geomspace(0.15, 1e-4, 12),
    [1.0],
    1.0 + np.geomspace(1e-4, 0.25, 8),
    np.geomspace(1.5, 2.8, 4)]))


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def station(s, seed):
    gk = s * K_CRIT
    mm = make_mass_model(gk, ETA)
    xyz = np.asarray(mass_xyz(mm, THETA0, seed, ETA))
    if not np.isfinite(xyz).all():
        raise FloatingPointError("non-finite equilibrium")
    xf = free_of(xyz)
    V = enclosed_volume_np(xyz, faces_tri)
    p = gk / V

    q = jnp.exp(THETA0).astype(QDTYPE)
    K = mm.stiffness_matrix(q, structure)
    R_fixed = mm.residual_fixed_matrix(q, bp.xyz_fixed, structure)
    Kd = np.asarray(K.todense())
    BIG = np.kron(np.linalg.inv(Kd), np.eye(3))

    def dPdx(model, loads):
        def fn(xr):
            return model.load_xyz_matrix_from_r_fixed(
                (K, R_fixed, bp.xyz_fixed, loads), xr.reshape(-1, 3), structure).ravel()
        return np.asarray(jax.jacrev(fn)(xf.ravel()))

    fm = _fixed_pressure_model(ETA)
    loads_f = bp.loads._replace(faces=UNIT_FACES * (SIGN * p))
    A_m = np.eye(N_DOF) - BIG @ dPdx(mm, bp.loads)            # gas law:   K^-1 K~_T
    A_f = np.eye(N_DOF) - BIG @ dPdx(fm, loads_f)             # prescribed: K^-1 K_T

    P_star = mm.load_xyz_matrix_from_r_fixed(
        (K, R_fixed, bp.xyz_fixed, bp.loads), xf, structure)
    residual = float(jnp.mean(jnp.linalg.norm(
        mm.linearsolve_fn(K, P_star) - xf, axis=1)))

    # rank-one factors and Sherman-Morrison. A_f - A_m is exactly rank one (one gas
    # chamber). Read the factors off the matrix instead of a full SVD (n^3), and report
    # how far the difference is from rank one.
    Dm = A_f - A_m
    j = int(np.argmax(np.linalg.norm(Dm, axis=0)))
    a = Dm[:, j].copy()
    i = int(np.argmax(np.abs(a)))
    b = Dm[i, :] / Dm[i, j]
    rank1 = float(np.linalg.norm(Dm - np.outer(a, b)) / np.linalg.norm(Dm))
    del Dm, BIG
    y = np.linalg.solve(A_m.T, b)
    delta = float(1.0 + a @ y)

    v = np.asarray(jax.grad(lambda xr: loss_from_xyz(
        mm.nodes_positions(xr.reshape(-1, 3), bp.xyz_fixed, structure)))(xf.ravel()))
    w_m = np.linalg.solve(A_m.T, v)
    w_f_sm = w_m - y * (a @ w_m) / delta
    try:
        w_f_lu = np.linalg.solve(A_f.T, v)
    except np.linalg.LinAlgError:
        w_f_lu = np.full_like(v, np.nan)

    def F_theta(th, xr):
        qq = jnp.exp(th).astype(QDTYPE)
        Kq = mm.stiffness_matrix(qq, structure)
        Rq = mm.residual_fixed_matrix(qq, bp.xyz_fixed, structure)
        Pq = mm.load_xyz_matrix_from_r_fixed(
            (Kq, Rq, bp.xyz_fixed, bp.loads), xr, structure)
        return mm.linearsolve_fn(Kq, Pq).ravel()

    _, vjp = jax.vjp(lambda th: F_theta(th, xf), THETA0)
    gref_m = np.asarray(vjp(jnp.asarray(w_m))[0])
    gref_f = np.asarray(vjp(jnp.asarray(w_f_sm))[0])
    glu_f = (np.asarray(vjp(jnp.asarray(w_f_lu))[0]) if np.isfinite(w_f_lu).all()
             else np.full_like(gref_f, np.nan))

    # jax_fdm adjoints
    g_m = np.asarray(jax.jit(jax.grad(
        lambda th: loss_from_xyz(mass_xyz(mm, th, xf, ETA))))(THETA0))

    def froz_xf(th):
        return fm.equilibrium_iterative_xyz(
            jnp.exp(th).astype(QDTYPE), bp.xyz_fixed, loads_f, structure,
            xyz_free_init=xf, tmax=TMAX, eta=ETA, solver=DAMPED, implicit_diff=True)

    drift = float(np.abs(np.asarray(froz_xf(THETA0)) - xf).max())
    g_f = np.asarray(jax.jit(jax.grad(lambda th: loss_from_xyz(
        fm.nodes_positions(froz_xf(th), bp.xyz_fixed, structure))))(THETA0))

    sm_vs_lu = rel(gref_f, glu_f) if np.isfinite(glu_f).all() else np.nan
    return dict(
        s=s, Vfrac=V / V_CRIT, pfrac=p / P_CRIT, residual=residual, drift=drift,
        smin_m=(float(np.linalg.svd(A_m, compute_uv=False).min()) if DENSE_SVD
                else np.nan),
        smin_f=(float(np.linalg.svd(A_f, compute_uv=False).min()) if DENSE_SVD
                else np.nan),
        delta=delta, rank1=rank1,
        err_m=rel(g_m, gref_m), err_m_probe=rel(g_m[PROBE], gref_m[PROBE]),
        err_f=rel(g_f, gref_f), err_f_probe=rel(g_f[PROBE], gref_f[PROBE]),
        sm_vs_lu=sm_vs_lu,
        ng_m=float(np.linalg.norm(gref_m)), ng_f=float(np.linalg.norm(gref_f)),
    ), xf


COLS = ("s", "Vfrac", "pfrac", "residual", "drift", "smin_m", "smin_f", "delta",
        "rank1",
        "err_m", "err_m_probe", "err_f", "err_f_probe", "sm_vs_lu", "ng_m", "ng_f")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 78)
    print(f"jaxfdm_adjoint_reference.py -- dense / Sherman-Morrison adjoint reference  [{CASE}, "
          f"tri{L.TRI_LEVEL}, {N_DOF} dof, {N_EDGES} edges]")
    print("=" * 78)
    print(f"[fold] SciPy: k_crit={K_CRIT:.6f}  V_crit={V_CRIT:.6f}  p_crit={P_CRIT:.6f}")
    print(f"{'s':>8} {'V/V*':>7} {'smin_m':>9} {'smin_f':>9} {'delta':>10} | "
          f"{'err mass':>9} {'err presc':>9} {'SM vs LU':>9} {'drift':>8}")
    rows, seed = [], free_of(L.vertices)
    for s in S:
        try:
            r, seed = station(float(s), seed)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{s:>8.5f}  FAILED {type(exc).__name__}: {exc}")
            continue
        if r["residual"] >= RES_TOL:
            print(f"{s:>8.5f}  residual {r['residual']:.1e} -- excluded")
            continue
        if r["drift"] > DRIFT_TOL:
            r["err_f"] = r["err_f_probe"] = np.nan        # left the state: not reported
        rows.append(r)
        np.savez(OUT, **{c: np.array([q[c] for q in rows], float) for c in COLS},
                 probe=PROBE, k_crit=K_CRIT, V_crit=V_CRIT, p_crit=P_CRIT,
                 n_dof=N_DOF, n_edges=N_EDGES, eta=ETA, drift_tol=DRIFT_TOL)
        print(f"{r['s']:>8.5f} {r['Vfrac']:>7.4f} {r['smin_m']:>9.2e} {r['smin_f']:>9.2e} "
              f"{r['delta']:>+10.2e} | {r['err_m']:>9.2e} {r['err_f']:>9.2e} "
              f"{r['sm_vs_lu']:>9.2e} {r['drift']:>8.1e}", flush=True)

    R = {c: np.array([q[c] for q in rows], float) for c in COLS}
    print(f"\n[saved] {OUT}  ({len(rows)} stations, {time.time() - t0:.0f} s)")
    pre, post = R["Vfrac"] < 1.0, R["Vfrac"] > 1.0
    print(f"gas-law adjoint vs dense   : median {np.median(R['err_m']):.2e}, "
          f"worst {R['err_m'].max():.2e}")
    ok = np.isfinite(R["err_f"])
    notfold = ok & (np.abs(R["Vfrac"] - 1.0) > 1e-6)
    print(f"prescribed adjoint vs S-M  : {ok.sum()} stations; far (s<0.75) median "
          f"{np.nanmedian(R['err_f'][R['s'] < 0.75]):.2e}; worst excluding V_f itself "
          f"{np.nanmax(R['err_f'][notfold]):.2e}; at V_f {np.nanmax(R['err_f'][ok & ~notfold]) if (ok & ~notfold).any() else np.nan:.2e}")
    k = np.isfinite(R["sm_vs_lu"]) & pre
    print(f"S-M vs dense LU (pre-fold) : worst {R['sm_vs_lu'][k].max():.2e}")
    print(f"A_f - A_m rank one         : worst residual {np.nanmax(R['rank1']):.2e}")
    x, yv = 1.0 / R["smin_f"][ok], R["err_f"][ok]
    k = np.isfinite(x)
    sl = (np.polyfit(np.log10(x[k]), np.log10(yv[k]), 1)[0] if k.sum() > 2
          else np.nan)
    print(f"prescribed err vs 1/sigma_min(A_f): log-log slope {sl:.2f}  (kappa*tol -> 1)")
