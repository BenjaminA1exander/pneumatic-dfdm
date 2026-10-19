r"""continuation.py -- volume-controlled continuation, spectra, and fold location.

WHY VOLUME CONTROL
    Under prescribed pressure the branch p(V) simply stops at its maximum: the
    Newton tangent K_T is singular there and there is no equilibrium beyond. Taking
    the enclosed volume as the control parameter and p as an unknown removes that.
    The unknowns are (x_u, p) and the equations are

        r(x_u, p) = 0,        V(x_u) - Vbar = 0

    with the bordered Jacobian

        J_b = [[ K_T, -grad V ], [ grad V^T, 0 ]]

    which stays NON-singular at a simple fold even though its (1,1) block does not.
    That is what lets the branch be traced straight through the pressure maximum.

THE LIMIT POINT, AND WHY THE NULL VECTOR MOVES VOLUME
    Differentiating the two equations with respect to Vbar gives the branch tangent

        J_b [x'; p'] = [0; 1].

    At a pressure maximum p' = 0, and the first block row collapses to K_T x' = 0
    while the second still reads grad V^T x' = 1. So at any interior maximum of p
    along the branch, K_T is singular AND its null vector has non-zero volume work
    -- not as an observation but as an identity. The measured quantities below
    (mu1 -> 0, and the normalised volume work of phi1 staying well away from zero)
    are the numerical expression of that.

SPECTRA
    K_T phi = mu M phi with M = kron(diag(m_i), I3), m_i one third of the reference
    triangle areas at node i. Since M is diagonal and positive, this is solved as the
    standard symmetric problem for A = M^(-1/2) K_T M^(-1/2), whose eigenvalues are
    identical and whose eigenvectors map back as phi = M^(-1/2) psi. That is both
    faster and better conditioned than handing ARPACK a generalised problem, and it
    is cross-checked against a dense generalised solve on the coarsest mesh.

    mu carries units of q / length^2, so the reported quantity is mu R^2 / q. At
    p = 0 and uniform q, kron(K,I3) = N0 * (cotangent Laplacian) on the equilateral
    interior, so mu1 R^2 / N0 must approach the first Dirichlet eigenvalue of the
    unit disk, j01^2 = 5.7832, with multiplicity three (one per coordinate).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigsh

_ZERO11 = sp.csr_matrix((1, 1))


def bordered(KT, gV):
    """J_b = [[K_T, -grad V], [grad V^T, 0]]."""
    return sp.bmat([[KT, -gV.reshape(-1, 1)],
                    [gV.reshape(1, -1), _ZERO11]], format="csc")


def newton(model, xu, p, Vbar, cfg):
    """Corrector. Returns (ok, xu, p, iters, state) with state carrying the
    operators already evaluated at the converged point."""
    tol_r = cfg.tol_r * cfg.q * cfg.R
    tol_v = cfg.tol_v * cfg.V_star
    xu, p = xu.copy(), float(p)
    for it in range(cfg.newton_max + 1):
        gV = model.grad_V(xu)
        V = model.volume(xu)
        r = model.residual(xu, p, gV)
        f2 = V - Vbar
        if np.linalg.norm(r) < tol_r and abs(f2) < tol_v:
            H = model.hess_V(xu)
            return True, xu, p, it, dict(V=V, gV=gV, H=H,
                                         KT=model.tangent(xu, p, H),
                                         res=float(np.linalg.norm(r)),
                                         verr=float(abs(f2)))
        if it == cfg.newton_max:
            break
        KT = model.tangent(xu, p)
        d = spsolve(bordered(KT, gV), -np.r_[r, f2])
        if not np.all(np.isfinite(d)):
            break
        xu = xu + d[:-1]
        p = p + d[-1]
    return False, xu, p, cfg.newton_max, None


def tangent_dir(state):
    """Branch tangent: J_b [x'; p'] = [0; 1]. Returns (xdot, pdot = dp/dVbar)."""
    n = state["KT"].shape[0]
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    z = spsolve(bordered(state["KT"], state["gV"]), rhs)
    return z[:-1], float(z[-1])


def spectra(model, state, cfg, n_eig=None, dense=False):
    """Lowest eigenvalues of K_T phi = mu M phi, plus the phi1 diagnostics."""
    n_eig = n_eig or cfg.n_eig
    KT, gV = state["KT"], state["gV"]
    s = 1.0 / np.sqrt(model.m_free)
    S = sp.diags(s)
    A = (S @ KT @ S).tocsc()

    if dense:
        from scipy.linalg import eigh
        w, Psi = eigh(A.toarray())
        mu, Psi = w[:n_eig], Psi[:, :n_eig]
    else:
        sigma = cfg.eig_shift * cfg.q / cfg.R ** 2
        mu, Psi = eigsh(A, k=n_eig, sigma=sigma, which="LM",
                        ncv=max(2 * n_eig + 1, 20))
        order = np.argsort(mu)
        mu, Psi = mu[order], Psi[:, order]

    phi1 = s * Psi[:, 0]
    if float(gV @ phi1) < 0.0:            # orient the mode so that it inflates
        phi1 = -phi1
    work = abs(float(gV @ phi1)) / (np.linalg.norm(gV) * np.linalg.norm(phi1))

    nrm = model.nodal_normals(state["xu"])[model.free]
    comp = np.einsum("ij,ij->i", phi1.reshape(-1, 3), nrm)
    z = model.full_xyz(state["xu"])[model.free, 2] / cfg.R
    corr = _pearson(comp, z)
    return mu, phi1, work, corr


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0.0 else float("nan")


def _record(model, cfg, xu, p, Vbar, state, iters, dV, pdot, eig=None):
    X = model.full_xyz(xu)
    z_apex = float(X[model.mesh["apex"], 2])
    row = dict(Vbar=float(Vbar), V=state["V"], V_over_Vstar=state["V"] / cfg.V_star,
               p=float(p), pR_over_N0=float(p) * cfg.R / cfg.N0,
               z_apex=z_apex, z_apex_over_R=z_apex / cfg.R,
               z_max=float(X[:, 2].max()),
               theta_h=2.0 * np.arctan(z_apex / cfg.R),
               dpdV=float(pdot), newton_iters=int(iters), dV=float(dV),
               res=state["res"], verr=state["verr"])
    if eig is None:
        for k in range(cfg.n_eig):
            row[f"mu{k + 1}"] = float("nan")
        row["volwork_phi1"] = float("nan")
        row["corr_phi1_z"] = float("nan")
    else:
        mu, _phi1, work, corr = eig
        for k in range(cfg.n_eig):
            row[f"mu{k + 1}"] = float(mu[k]) * cfg.R ** 2 / cfg.q if k < len(mu) \
                else float("nan")
        row["volwork_phi1"] = float(work)
        row["corr_phi1_z"] = float(corr)
    return row


def sweep(model, cfg, log=print):
    """Trace the branch from the flat p = 0 state to v_end_frac * V*.

    Returns (rows, states) where states[i] carries the full converged data of
    row i -- needed by the fold bisection and by the figure."""
    Vstar = cfg.V_star
    xu = model.reference().copy()
    p = 0.0
    Vbar = 0.0

    ok, xu, p, iters, st = newton(model, xu, p, Vbar, cfg)
    if not ok:
        raise RuntimeError("flat reference state did not converge")
    st["xu"] = xu
    xdot, pdot = tangent_dir(st)
    eig = spectra(model, st, cfg)
    rows = [_record(model, cfg, xu, p, Vbar, st, iters, 0.0, pdot, eig)]
    states = [dict(Vbar=Vbar, xu=xu.copy(), p=p, pdot=pdot, xdot=xdot,
                   V=st["V"], mu=eig[0].copy())]

    dV = cfg.v_start_frac * Vstar
    dV_max = cfg.dv_frac * Vstar
    dV_min = cfg.dv_min_frac * Vstar
    V_end = cfg.v_end_frac * Vstar
    k = 0
    while Vbar < V_end - 1e-12 * Vstar:
        step = min(dV, V_end - Vbar)
        ok, xn, pn, iters, stn = newton(
            model, xu + step * xdot, p + step * pdot, Vbar + step, cfg)
        if not ok:
            dV *= 0.5
            if dV < dV_min:
                log(f"    [stop] step collapsed below dV_min at V/V* = "
                    f"{Vbar / Vstar:.4f}")
                break
            continue
        xu, p, Vbar = xn, pn, Vbar + step
        stn["xu"] = xu
        xdot, pdot = tangent_dir(stn)
        k += 1
        eig = spectra(model, stn, cfg) if (k % cfg.eig_stride == 0) else None
        rows.append(_record(model, cfg, xu, p, Vbar, stn, iters, step, pdot, eig))
        states.append(dict(Vbar=Vbar, xu=xu.copy(), p=p, pdot=pdot, xdot=xdot,
                           V=stn["V"], mu=(eig[0].copy() if eig else None)))
        if iters > cfg.newton_shrink:
            dV *= 0.5
        elif iters <= cfg.newton_grow:
            dV = min(dV * 1.5, dV_max)
    return rows, states


def locate_fold(model, cfg, states, log=print):
    """Bisect on Vbar for the sign change of p' = dp/dVbar."""
    pd = np.array([s["pdot"] for s in states])
    idx = np.where((pd[:-1] > 0.0) & (pd[1:] < 0.0))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    lo, hi = states[i], states[i + 1]
    log(f"    fold bracketed by V/V* = {lo['Vbar'] / cfg.V_star:.5f} "
        f"(p' = {lo['pdot']:+.4e}) and {hi['Vbar'] / cfg.V_star:.5f} "
        f"(p' = {hi['pdot']:+.4e})")

    best = None
    for it in range(cfg.fold_max_bisect):
        Vm = 0.5 * (lo["Vbar"] + hi["Vbar"])
        w = (Vm - lo["Vbar"]) / (hi["Vbar"] - lo["Vbar"])
        xg = (1.0 - w) * lo["xu"] + w * hi["xu"]
        pg = (1.0 - w) * lo["p"] + w * hi["p"]
        ok, xm, pm, iters, st = newton(model, xg, pg, Vm, cfg)
        if not ok:
            log(f"    [fold] corrector failed at V/V* = {Vm / cfg.V_star:.8f}")
            break
        st["xu"] = xm
        xdot, pdot = tangent_dir(st)
        best = dict(Vbar=Vm, xu=xm.copy(), p=pm, pdot=pdot, xdot=xdot,
                    V=st["V"], state=st, n_bisect=it + 1)
        if abs(pdot) < cfg.fold_tol:
            break
        if hi["Vbar"] - lo["Vbar"] < 1e-14 * cfg.V_star:
            break
        if pdot > 0.0:
            lo = dict(Vbar=Vm, xu=xm, p=pm, pdot=pdot)
        else:
            hi = dict(Vbar=Vm, xu=xm, p=pm, pdot=pdot)

    if best is None:
        return None
    st = best["state"]
    n_eig = max(cfg.n_eig, 4)
    mu, phi1, work, corr = spectra(model, st, cfg, n_eig=n_eig)
    X = model.full_xyz(best["xu"])
    z_apex = float(X[model.mesh["apex"], 2])
    best.update(mu=mu, phi1=phi1, volwork=work, corr=corr,
                z_apex=z_apex, theta_h=2.0 * np.arctan(z_apex / cfg.R),
                xdot_volwork=abs(float(st["gV"] @ best["xdot"]))
                / (np.linalg.norm(st["gV"]) * np.linalg.norm(best["xdot"])))
    return best
