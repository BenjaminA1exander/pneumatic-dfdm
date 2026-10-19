r"""cap_verify.py -- the checks of Sec. 3.1 on one cap and one mesh: Table 1 and Fig. 3(d).

Same model as cap_model.py (imported, not copied). The script locates the limit point,
drives prescribed pressure towards it, evaluates every quantity of Eq. 22 on both
sides of it, and runs the gas-law continuation through it. In the order of the paper:

 1. FOLD      volume-controlled path (Eq. 10) to the pressure maximum; the maximum is
              bracketed by the sign of dp/dVbar (branch tangent) and bisected to a
              bracket of 1e-13 V* -> (V_f, p_f). There lambda_min(K^-1 K_T) is zero to
              that precision (claim i) and lambda_min(K^-1 K~_T) is the gas-law entry of
              Table 1 (claim iii). FOLD_ONLY=1 stops here and writes fold_<tag>.npz
              (the limit-point convergence of Fig. 3e, Table 1 rows p_f/p* and lambda_min).
 2. P SWEEP   prescribed-pressure Newton on the stable branch at p = p_f (1 - eps),
              eps = 1e-1 ... 1e-7, started from the volume-path state with the nearest
              pressure below; and at p_f (1 + eps), where no equilibrium exists and
              Newton must fail (claim ii: nothing beyond the limit point).
 3. AT EACH STATE (P sweep, and volume-path states past the fold at V/V_f - 1 = 1e-4 ... 0.6,
              the branch that is unstable under prescribed pressure):
       S_solve = (p/V) g^T K_T^-1 g                    S of Eq. 22, the quantity in the paper
       S_tan   = (p/V) / (dp/dVbar)                    from the bordered tangent: another system
       S_fd    = [ln V(p e^d) - ln V(p e^-d)] / 2d     two extra prescribed-pressure Newton solves
       dx_P    = p K_T^-1 g,  dx_G = p K~_T^-1 g        responses to ln p and to ln K_gas, each
                                                       checked by central differences of the
                                                       nonlinear solve (claim v)
       ||dx_P - (1 + S) dx_G|| / ||dx_P||              the vector identity of Eq. 22
       cP / cG against (1 + S)^2                        Gauss-Newton curvatures of L = mean z_i^2
                                                       along ln p / ln K_gas (Fig. 3d, Table 1)
       |dL/dtheta| under both closures                  L = n^-1 sum (z_i - 0.8 R)^2, theta = ln q,
                                                       adjoint of each closure; their ratio is the
                                                       "gradient ratio" of Table 1 (claim ii)
       negative eigenvalue counts of (K_T, K), (K~_T, K): 10 lowest by shift-invert, and DENSE
                                                       at h/R <= 1/12 (claim iii)
 4. GAS LAW   continuation in K_gas (gas-law Newton, no volume control) from 0.8 V_f to
              1.6 V_f: passes the fold without a jump in V and with neg(K~_T) = 0, in at
              most a few Newton iterations per step (claim iii).
 5. POLE      |dx/dln p| and |dL/dtheta| ~ (1 - V/V_f)^n fitted over 1e-4 < 1 - V/V_f < 1e-2:
              n = -1 under prescribed pressure (simple pole in V), n = 0 under the gas law.
 6. GRAD FD   both adjoint gradients against a directional finite difference of L(theta)
              at fixed p (prescribed) and fixed K_gas (gas law), at the first P-sweep state.

The [table1] block at the end prints this mesh's column of Table 1.

RUN   CAP=uniform TRI_LEVEL=6  python cap_verify.py          (63 s on a laptop)
      CAP=zoned   TRI_LEVEL=24 python cap_verify.py          (~4 min)
      FOLD_ONLY=1 CAP=uniform TRI_LEVEL=96 python cap_verify.py   (limit point only, finer meshes)
ENV   as cap_model.py, plus FOLD_ONLY
OUT   results/sec31/verify_<tag>.npz  or  fold_<tag>.npz  (new file per run) + stdout
"""
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import cap_model as B                                                   # noqa: E402
from fdm_limit_point.continuation import newton as vc_newton, tangent_dir  # noqa: E402

model, cfg, VS, N, K3 = B.model, B.cfg, B.VS, B.N, B.K3
TOL = cfg.tol_r * cfg.q * cfg.R
D = 1e-5                                                 # relative step of the FD checks
t0 = time.time()
ZD = np.arange(2, N, 3)                                  # z dofs of the free nodes
HL = 2.0 / (N // 3)                                      # Hessian of L = mean z_i^2 on the z dofs


def log(s):
    print(s, flush=True)


def newton_p(x, p, nmax=40):
    """Prescribed-pressure Newton: K_T dx = -r at fixed p (Eq. 5)."""
    x = x.copy()
    for it in range(nmax + 1):
        r = model.residual(x, p)
        nr = np.linalg.norm(r)
        if nr < TOL:
            return True, x, it
        if it == nmax or not np.isfinite(nr) or nr > 1e6:
            break
        x = x + spsolve(model.tangent(x, p), -r)
    return False, x, nmax


def newton_K(x, Kg, nmax=40):
    """Gas-law Newton: K~_T dx = -r with p = K_gas / V(x) (Eqs. 16-18)."""
    x = x.copy()
    for it in range(nmax + 1):
        V = model.volume(x)
        p = Kg / V
        r = model.residual(x, p)
        nr = np.linalg.norm(r)
        if nr < TOL:
            return True, x, it
        if it == nmax or not np.isfinite(nr) or nr > 1e6 or V <= 0:
            break
        g = model.grad_V(x)
        x = x + solve_KTg(model.tangent(x, p), g, p, V, -r)
    return False, x, nmax


def solve_KTg(KT, g, p, V, b):
    """K~_T z = b through the bordered matrix [[K_T, g], [g^T, -V/p]] (Eqs. 19-20);
    no Sherman-Morrison, no dense outer product."""
    A = sp.bmat([[KT, g.reshape(-1, 1)], [g.reshape(1, -1), sp.csr_matrix([[-V / p]])]], format="csc")
    return spsolve(A, np.r_[b, 0.0])[:-1]


def counts(KT, g, p, V, dense):
    """Negative eigenvalue counts of (K_T, K) and (K~_T, K): sparse, and dense where affordable."""
    wP, _ = B.lowest(KT, k=10)
    wG, _ = B.lowest(KT, p / V, g, k=10)
    out = [int((wP < 0).sum()), int((wG < 0).sum())]
    if dense:
        Kd = K3.toarray()
        dP = sla.eigh(KT.toarray(), Kd, eigvals_only=True, subset_by_index=[0, 9])
        dG = sla.eigh(KT.toarray() + (p / V) * np.outer(g, g), Kd, eigvals_only=True, subset_by_index=[0, 9])
        out += [int((dP < 0).sum()), int((dG < 0).sum()), float(dP[0]), float(dG[0])]
    else:
        out += [-1, -1, float("nan"), float("nan")]
    return out + [float(wP[0]), float(wG[0])]


ZT = 0.8 * cfg.R                                          # objective: L = n^-1 sum (z_i - 0.8 R)^2
NU = N // 3


def dL_dx(x):
    gx = np.zeros(N)
    gx[ZD] = 2.0 * (x[ZD] - ZT) / NU
    return gx


def objective(x):
    return float(np.sum((x[ZD] - ZT) ** 2) / NU)


def theta_grad(x, lam):
    """dL/dtheta_e = -lam^T dr/dtheta_e, theta = ln q:  dr/dtheta_e = q_e Cu_e^T (C X)_e  (per coordinate)."""
    X = model.full_xyz(x)
    edge_vec = B.Cm @ X                                   # (m, 3): x_i - x_j over each edge
    Lam = lam.reshape(-1, 3)
    return -B.q * np.sum((B.Cu @ Lam) * edge_vec, axis=1)


def set_q(qv):
    Qd = sp.diags(qv)
    model.K = (B.Cu.T @ Qd @ B.Cu).tocsc()
    model.D = (B.Cu.T @ Qd @ B.Cs).tocsc()
    model.K3 = sp.kron(model.K, sp.eye(3), format="csc")
    model.D3 = sp.kron(model.D, sp.eye(3), format="csc")


def gradient_fd_check(x, p):
    """Directional FD of L(theta) at fixed p (prescribed) and fixed K_gas (gas law) against the adjoints."""
    V = model.volume(x)
    g = model.grad_V(x)
    KT = model.tangent(x, p)
    gx = dL_dx(x)
    gP = theta_grad(x, spsolve(KT, gx))
    gG = theta_grad(x, solve_KTg(KT, g, p, V, gx))
    rng = np.random.default_rng(0)
    dth = rng.standard_normal(len(B.q))
    dth /= np.linalg.norm(dth)
    out = []
    for kind, gr in (("P", gP), ("G", gG)):
        vals = []
        for t in (1e-5, -1e-5):
            set_q(B.q * np.exp(t * dth))
            ok, xt, _ = newton_p(x, p) if kind == "P" else newton_K(x, p * V)
            vals.append(objective(xt) if ok else np.nan)
        set_q(B.q)
        fd = (vals[0] - vals[1]) / 2e-5
        out.append((kind, float(gr @ dth), fd))
    return out


def analyse(x, p, label, dense, d=D):
    """Every per-state quantity of step 3 of the docstring."""
    V = model.volume(x)
    g = model.grad_V(x)
    KT = model.tangent(x, p)
    yP = spsolve(KT, g)
    S_solve = (p / V) * float(g @ yP)
    ok_v, xv, pv, _, st = vc_newton(model, x, p, V, cfg)            # same state, gives the bordered operators
    _, pdot = tangent_dir(st)
    S_tan = (p / V) / pdot
    okp, xp_, _ = newton_p(x, p * np.exp(d))
    okm, xm_, _ = newton_p(x, p * np.exp(-d))
    S_fd = (np.log(model.volume(xp_)) - np.log(model.volume(xm_))) / (2 * d) if okp and okm else np.nan
    dxP = p * yP
    dxP_fd = (xp_ - xm_) / (2 * d) if okp and okm else np.full(N, np.nan)
    dxG = p * solve_KTg(KT, g, p, V, g)
    Kg = p * V
    okKp, xKp, _ = newton_K(x, Kg * np.exp(D))
    okKm, xKm, _ = newton_K(x, Kg * np.exp(-D))
    dxG_fd = (xKp - xKm) / (2 * D) if okKp and okKm else np.full(N, np.nan)
    e_P = np.linalg.norm(dxP - dxP_fd) / np.linalg.norm(dxP)
    e_G = np.linalg.norm(dxG - dxG_fd) / np.linalg.norm(dxG)
    e_id = np.linalg.norm(dxP - (1 + S_solve) * dxG) / np.linalg.norm(dxP)
    cP = HL * float(dxP[ZD] @ dxP[ZD])
    cG = HL * float(dxG[ZD] @ dxG[ZD])
    c = counts(KT, g, p, V, dense)
    gx = dL_dx(x)
    gradP = theta_grad(x, spsolve(KT, gx))
    gradG = theta_grad(x, solve_KTg(KT, g, p, V, gx))
    row = dict(label=label, V=V, p=p, S_solve=S_solve, S_tan=S_tan, S_fd=S_fd, e_dxP_fd=e_P, e_dxG_fd=e_G,
               e_identity=e_id, cP=cP, cG=cG, ratio=cP / cG, onepS2=(1 + S_solve) ** 2, nP=c[0], nG=c[1],
               nP_dense=c[2], nG_dense=c[3], lamP_dense=c[4], lamG_dense=c[5], lamP=c[6], lamG=c[7],
               dxP_norm=float(np.linalg.norm(dxP)), gradP=float(np.linalg.norm(gradP)),
               gradG=float(np.linalg.norm(gradG)), resid=float(np.linalg.norm(model.residual(x, p))))
    return row


def show(r, Vf):
    log(f"{r['label']:>14} {r['V'] / Vf - 1:+11.3e} {r['p']:10.6f} {r['S_solve']:12.5f} {r['S_tan']:12.5f} "
        f"{r['S_fd']:12.5f} {r['e_dxP_fd']:9.1e} {r['e_dxG_fd']:9.1e} {r['e_identity']:9.1e} {r['ratio']:12.5e} "
        f"{r['onepS2']:12.5e} {r['nP']:3d}/{r['nP_dense']:2d} {r['nG']:3d}/{r['nG_dense']:2d} {r['lamP']:+10.3e} "
        f"{r['lamG']:+10.3e} {r['dxP_norm']:10.3e} {r['gradP']:10.3e} {r['gradG']:10.3e}")


def table1_column(rows, Vf, pf, lf_P, lf_G, poles):
    """This mesh's column of Table 1, from the quantities just computed."""
    by = {r["label"]: r for r in rows}
    log(f"\n[table1] {B.CAP} cap, h/R = 1/{B.LEVEL}  (gradient ratio = |dL/dtheta|_p / |dL/dtheta|_Kgas at p = p_f(1 - eps))")
    log(f"[table1] p_f/p*                       {pf / cfg.p_star:.4f}")
    log(f"[table1] lambda_min at V_f, prescribed {lf_P[0]:.1e}   gas law {lf_G[0]:.3f}   (vol. constraint: cap_closure_probe.py)")
    for eps in ("1e-03", "1e-07"):
        r = by.get(f"pre {eps}")
        if r:
            log(f"[table1] eps = {eps}: gradient ratio {r['gradP'] / r['gradG']:.1f}   (1 + S)^2 {r['onepS2']:.4g}   "
                f"(S = {r['S_solve']:.2f}; cP/cG = {r['ratio']:.6g})")
    for name, n in poles.items():
        log(f"[table1] pole exponent n, {name}: {n:+.4f}")


def main():
    dense = B.LEVEL <= 12
    log(f"[verify] {B.CAP} cap  h/R = 1/{B.LEVEL}  dof {N}  GAMMA {B.GAMMA} BETA {B.BETA} RB {B.RB}  dense eigen checks: {dense}")
    # ---- 1. volume path to the fold, bracket and bisect the pressure maximum
    x, p, Vb = model.x_flat.copy(), 0.0, 0.0
    path = []
    dV = 0.01 * VS
    prev = None
    while True:
        ok, x1, p1, it, st = vc_newton(model, x, p, Vb + dV, cfg)
        if not ok:
            dV *= 0.5
            assert dV > 1e-8 * VS, "continuation failed before the fold"
            continue
        x, p, Vb = x1, p1, Vb + dV
        _, pdot = tangent_dir(st)
        path.append((Vb, p, x.copy(), pdot))
        if prev is not None and prev[3] > 0 and pdot < 0:
            break
        prev = path[-1]
    (Va, pa, xa, da), (Vc, pc, xc, dc) = path[-2], path[-1]
    for _ in range(60):
        Vm = 0.5 * (Va + Vc)
        ok, xm, pm, it, st = vc_newton(model, xa, pa, Vm, cfg)
        _, dm = tangent_dir(st)
        if dm > 0:
            Va, pa, xa, da = Vm, pm, xm, dm
        else:
            Vc, pc, xc, dc = Vm, pm, xm, dm
        if Vc - Va < 1e-13 * VS:
            break
    Vf, pf, xf = Va, pa, xa
    gfold = model.grad_V(xf)
    KTf = model.tangent(xf, pf)
    lf_P, _ = B.lowest(KTf, k=2)
    lf_G, _ = B.lowest(KTf, pf / model.volume(xf), gfold, k=2)
    log(f"[fold] lambda_min(K^-1 K_T) = {lf_P[0]:+.3e}, lambda_min(K^-1 K~_T) = {lf_G[0]:.6f}, "
        f"lambda_2(K^-1 K_T) = {lf_P[1]:.6f}")
    log(f"[fold] V_f/V* = {Vf / VS:.10f}  p_f/p* = {pf / cfg.p_star:.10f}  bracket {Vc - Va:.1e}  dp/dV {da:+.2e}/{dc:+.2e}")
    if os.environ.get("FOLD_ONLY", "0") == "1":                # fold location and margins only (mesh convergence)
        tag = f"fold_h{B.LEVEL}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_{time.strftime('%Y%m%d-%H%M%S')}"
        Xf = model.full_xyz(xf)
        z_apex = float(Xf[B.mesh["apex"], 2])
        log(f"[fold] apex height z_f/R = {z_apex:.10f}")
        np.savez(os.path.join(B.OUT, tag + ".npz"), Vf=Vf, pf=pf, z_apex=z_apex, lamP_fold=lf_P, lamG_fold=lf_G, n_dof=N)
        log(f"[verify] FOLD_ONLY done in {time.time() - t0:.0f} s -> {tag}.npz")
        return

    # V path past the fold (unstable branch), for analysis and for P starts
    post = []
    x, p = xc, pc
    for s in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6]:
        ok, x, p, it, st = vc_newton(model, x, p, Vf * (1 + s), cfg)
        assert ok, f"V path failed at {s}"
        post.append((s, x.copy(), p))

    hdr = (f"{'state':>14} {'V/V_f - 1':>11} {'p':>10} {'S solve':>12} {'S tangent':>12} {'S fin.diff':>12} "
           f"{'dxP FD':>9} {'dxG FD':>9} {'identity':>9} {'cP/cG':>12} {'(1+S)^2':>12} {'nP s/d':>6} {'nG s/d':>6} "
           f"{'lam1 KT':>10} {'lam1 KTg':>10} {'|dx/dlnp|':>10} {'|dL/dth| P':>10} {'|dL/dth| G':>10}")
    # ---- 2. prescribed pressure driven up to the fold
    log("\n[P sweep] prescribed-pressure Newton at p = p_f (1 - eps), stable branch")
    log(hdr)
    rows = []
    Vs = np.array([q[0] for q in path])
    # stable-branch starts: p = p_f (1 - A u^2) near the fold, A from one volume-controlled state at u = 1e-2
    ok, xA, pA, _, _ = vc_newton(model, xf, pf, Vf * (1 - 1e-2), cfg)
    A = (1 - pA / pf) / 1e-4
    log(f"[P sweep] fold curvature A = {A:.4f} (p = p_f (1 - A u^2), u = 1 - V/V_f); starts at u = sqrt(eps / A)")
    for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]:
        pt = pf * (1 - eps)
        u0 = min(np.sqrt(eps / A), 0.5)
        ok0, xs0, _, _, _ = vc_newton(model, xf, pf, Vf * (1 - u0), cfg)
        ok, x, it = newton_p(xs0, pt) if ok0 else (False, xf, 0)
        if not ok:
            log(f"{'eps ' + format(eps, '.0e'):>14}  prescribed-pressure Newton FAILED")
            continue
        if model.volume(x) >= Vf:
            log(f"{'eps ' + format(eps, '.0e'):>14}  converged to the unstable branch (V > V_f): start not close enough")
        r = analyse(x, pt, f"pre {eps:.0e}", dense, d=min(D, 0.1 * eps))
        r["x"] = x.copy()
        r["newton_its"] = it
        rows.append(r)
        show(r, Vf)
    for eps in [1e-6, 1e-4, 1e-2]:
        ok, x, it = newton_p(xf, pf * (1 + eps))
        log(f"{'above ' + format(eps, '.0e'):>14}  p = p_f (1 + {eps:.0e}): Newton {'CONVERGED (!)' if ok else 'fails'}"
            f" after {it} its, V/V_f - 1 = {model.volume(x) / Vf - 1:+.3e}")
    log("\n[post fold] volume-path states beyond the fold (unstable branch under prescribed pressure)")
    log(hdr)
    for s, x, p in post:
        r = analyse(x, p, f"post {s:.0e}", dense)
        rows.append(r)
        show(r, Vf)

    # ---- 4. gas-law continuation in K_gas through the fold
    log("\n[gas law] continuation in K_gas, gas-law Newton only")
    i0 = np.argmin(np.abs(Vs - 0.8 * Vf))
    x = path[i0][2]
    K0 = path[i0][1] * path[i0][0]
    xe = post[-2][1]
    K1 = post[-2][2] * model.volume(xe)
    gas = []
    Ks = np.linspace(K0, max(K1, 1.6 * K0), 400)
    for k, Kg in enumerate(Ks):
        ok, x, it = newton_K(x, Kg)
        if not ok:
            log(f"[gas law] Newton FAILED at K/K0 = {Kg / K0:.5f}")
            break
        V = model.volume(x)
        p = Kg / V
        g = model.grad_V(x)
        KT = model.tangent(x, p)
        S = (p / V) * float(g @ spsolve(KT, g))
        if k % 20 == 0 or k == len(Ks) - 1 or (gas and (gas[-1][1] - Vf) * (V - Vf) <= 0):
            wG, _ = B.lowest(KT, p / V, g, k=6)
            wP, _ = B.lowest(KT, k=6)
            nG, nP = int((wG < 0).sum()), int((wP < 0).sum())
        else:
            nG = nP = -1
        gas.append((Kg, V, p, S, nP, nG, it))
        if nG >= 0:
            log(f"   K/K0 {Kg / K0:8.5f}  V/V_f {V / Vf:9.6f}  p/p_f {p / pf:9.6f}  S {S:11.4f}  "
                f"neg K_T {nP}  neg K~_T {nG}  newton {it}")
    gas = np.array(gas)
    dVs = np.diff(gas[:, 1])
    log(f"[gas law] {len(gas)} states, V monotone in K: {bool(np.all(dVs > 0))}, max relative V jump per step "
        f"{np.max(dVs / gas[:-1, 1]):.3e} (mean {np.mean(dVs / gas[:-1, 1]):.3e}), min S {gas[:, 3].min():.4f}, "
        f"max S past the fold {gas[gas[:, 1] > Vf, 3].max() if np.any(gas[:, 1] > Vf) else np.nan:.4f}, "
        f"Newton its max {int(gas[:, 6].max())}")

    # ---- 5. pole exponents on the stable branch
    pre = [r for r in rows if r["label"].startswith("pre")]
    u = np.array([1 - r["V"] / Vf for r in pre])
    sel = (u > 1e-4) & (u < 1.05e-2)                      # the window of Table 1
    poles = {}
    for key, name in (("dxP_norm", "|dx/dln p|"), ("gradP", "|dL/dtheta| prescribed"), ("gradG", "|dL/dtheta| gas law")):
        nrm = np.array([r[key] for r in pre])
        if sel.sum() >= 3:
            n_exp = np.polyfit(np.log(u[sel]), np.log(nrm[sel]), 1)[0]
            poles[name] = n_exp
            log(f"[pole] {name} ~ (1 - V/V_f)^n, 1e-4 <= 1 - V/V_f <= 1.05e-2: n = {n_exp:.4f} ({sel.sum()} points)")
    # ---- 6. adjoint gradients against finite differences of the nonlinear solve
    for kind, adj, fd in gradient_fd_check(pre[0]["x"], pre[0]["p"]) if pre and "x" in pre[0] else []:
        log(f"[grad FD] {kind}: adjoint {adj:+.10e}  FD {fd:+.10e}  rel {abs(adj - fd) / abs(fd):.1e}")
    tag = f"verify_h{B.LEVEL}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_{time.strftime('%Y%m%d-%H%M%S')}"
    keys = [k for k in rows[0] if k not in ("label", "x")]
    np.savez(os.path.join(B.OUT, tag + ".npz"), Vf=Vf, pf=pf, labels=np.array([r["label"] for r in rows]),
             keys=np.array(keys), table=np.array([[r.get(k, np.nan) for k in keys] for r in rows], dtype=float),
             gas=gas, gas_cols=np.array(["K", "V", "p", "S", "negKT", "negKTg", "newton"]))
    table1_column(rows, Vf, pf, lf_P, lf_G, poles)
    log(f"[verify] done in {time.time() - t0:.0f} s -> {tag}.npz")


if __name__ == "__main__":
    main()
