r"""cap_optimise_lib.py -- design of the cap under each closure: forward solvers, adjoint
gradients, stability check and the L-BFGS driver used for claim (vi) / Fig. 3(f).

All three closures are solved by Newton on the SAME discrete equations (Eq. 4)
    r(x, p) = kron(K,I3) x_u + kron(D,I3) x_s - p grad V(x) = 0
closed by
    prescribed pressure   p = p_k                         Newton matrix  K_T                 (Eq. 6)
    gas law               p = K_gas / V(x)                Newton matrix  K_T + (p/V) g g^T   (Eq. 18)
    volume constraint     V(x) = Vbar, p a multiplier     Newton matrix  [[K_T, -g], [g^T, 0]]
(g = grad V, K_T = kron(K,I3) - p grad^2 V, exact: fdm_limit_point.model). The gas-law
Newton matrix is applied through the equivalent bordered form [[K_T, g], [g^T, -V/p]]
(Eqs. 19-20), so no dense outer product is formed.

Design variables theta = ln q (one per edge). The gradient of an objective L(x) is the
adjoint of each closure's Newton system (prescribed: K_T, gas: K~_T, volume: J_b^T),
pulled back to theta through dr/dtheta_e = q_e C_e^T (C x)_e.

The optimiser is a plain L-BFGS (memory MEM, Armijo backtracking with constant C1, step
capped so that no theta_e moves by more than TRUST per iteration). A failed forward solve
counts as a failed trial step. Every 10 iterations the lowest eigenvalue of the closure's
own tangent is recorded (stability check).

The driver (cap_optimise_target.py) sets the objective (`loss`, `dL_dx`), the start
design (`Q0`, `TH0`) and the normalisers P_STAR, V_STAR of the history before calling
`optimise`. Nothing here reads files.

ENV   TRI_LEVEL (default 6): the mesh of the uniform-q model built at import
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve, eigsh, LinearOperator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fdm_limit_point.config import Config                      # noqa: E402
from fdm_limit_point import mesh as fmesh                      # noqa: E402
from fdm_limit_point.model import Model                        # noqa: E402
from fdm_limit_point.continuation import newton as vc_newton   # noqa: E402

RD = os.path.join(ROOT, "results")
LEVEL = int(os.environ.get("TRI_LEVEL", 6))
NEWTON_MAX = 50

cfg = Config()
cfg.newton_max = NEWTON_MAX
TOL_R = cfg.tol_r * cfg.q * cfg.R
Z_TARGET = 0.8 * cfg.R

mesh = fmesh.build(LEVEL, cfg)
model = Model(mesh, cfg, "uniform")
x_flat = model.reference()
K3 = model.K3                        # metric of the generalised eigenproblems (reset by the driver for a zoned start)
K3_lu = splu(K3.tocsc())
N = model.ndof
E = model.edges
FREE = model.free
APEX = mesh["apex"]

# normalisers of the recorded history (set by the driver to the cap's own limit point)
P_STAR = V_STAR = 1.0


# ----------------------------------------------------------------------------- Newton solvers
def newton_presc(x, p):
    x = x.copy()
    for it in range(NEWTON_MAX + 1):
        r = model.residual(x, p)
        nr = np.linalg.norm(r)
        if nr < TOL_R:
            return True, x, it
        if it == NEWTON_MAX or not np.isfinite(nr) or nr > 1e8:
            break
        dx = spsolve(model.tangent(x, p), -r)
        if not np.all(np.isfinite(dx)):
            break
        x = x + dx
    return False, x, NEWTON_MAX


def gas_matrix(x, Kg):
    V = model.volume(x)
    p = Kg / V
    g = model.grad_V(x)
    KT = model.tangent(x, p)
    B = sp.bmat([[KT, g.reshape(-1, 1)], [g.reshape(1, -1), sp.csr_matrix([[-V / p]])]],
                format="csc")
    return B, KT, g, V, p


def newton_gas(x, Kg):
    x = x.copy()
    for it in range(NEWTON_MAX + 1):
        V = model.volume(x)
        r = model.residual(x, Kg / V)
        nr = np.linalg.norm(r)
        if nr < TOL_R:
            return True, x, it
        if it == NEWTON_MAX or not np.isfinite(nr) or nr > 1e8 or V <= 0:
            break
        B = gas_matrix(x, Kg)[0]
        d = spsolve(B, -np.r_[r, 0.0])
        x = x + d[:-1]
    return False, x, NEWTON_MAX


def newton_vol(x, p, Vbar):
    ok, x, p, it, _ = vc_newton(model, x, p, Vbar, cfg)
    return ok, x, p, it


# ----------------------------------------------------------------------------- spectra
def k_op(n_extra=0, s0=None):
    if n_extra == 0:
        return (LinearOperator((N, N), matvec=lambda v: K3 @ v),
                LinearOperator((N, N), matvec=K3_lu.solve))
    M = LinearOperator((N + 1, N + 1), matvec=lambda v: np.r_[K3 @ v[:-1], s0 * v[-1]])
    Mi = LinearOperator((N + 1, N + 1),
                        matvec=lambda v: np.r_[K3_lu.solve(v[:-1]), v[-1] / s0])
    return M, Mi


def lowest_four(KT, c=0.0, g=None):
    """4 eigenvalues of (KT + c g g^T, K3) nearest -1 (= the four smallest here)."""
    A = (KT + K3).tocsc()                       # (KT + c gg^T) - (-1) K3
    lu = splu(A)
    if c == 0.0:
        opinv = LinearOperator((N, N), matvec=lu.solve)
        Aop = KT
    else:
        y = lu.solve(g)
        den = 1.0 + c * (g @ y)
        opinv = LinearOperator((N, N), matvec=lambda v: (lambda z: z - y * (c * (g @ z)) / den)(lu.solve(v)))
        Aop = LinearOperator((N, N), matvec=lambda v: KT @ v + c * g * (g @ v))
    M, _ = k_op()
    w = eigsh(Aop, k=4, M=M, sigma=-1.0, which="LM", OPinv=opinv, tol=1e-12,
              return_eigenvectors=False)
    return np.sort(w)


# ----------------------------------------------------------------------------- objective (default; the driver overrides)
def dL_dx(x):
    z = x.reshape(-1, 3)[:, 2]
    d = np.zeros((len(z), 3))
    d[:, 2] = 2.0 * (z - Z_TARGET) / len(z)
    return d.ravel()


def loss(x):
    z = x.reshape(-1, 3)[:, 2]
    return float(np.mean((z - 0.8 * cfg.R) ** 2))


# ----------------------------------------------------------------------------- design
m = len(E)
Cm = sp.coo_matrix((np.r_[np.ones(m), -np.ones(m)],
                    (np.r_[np.arange(m), np.arange(m)], np.r_[E[:, 0], E[:, 1]])),
                   shape=(m, mesh["n_nodes"])).tocsr()
Cu, Cs = Cm[:, model.free], Cm[:, model.fixed]
Q0 = model.q_edge.copy()
TH0 = np.log(Q0)
MAXIT, MEM, TRUST, C1 = 1000, 10, 0.1, 1e-4


def set_q(q):
    Qd = sp.diags(q)
    model.K3 = sp.kron(Cu.T @ Qd @ Cu, sp.eye(3), format="csc")
    model.D3 = sp.kron(Cu.T @ Qd @ Cs, sp.eye(3), format="csc")


def pullback(x, lam, q):
    """dL/dtheta = -lam^T dr/dtheta, with dr/dtheta_e = q_e C_e^T (C x)_e on the free rows."""
    X = model.full_xyz(x)
    L = np.zeros((mesh["n_nodes"], 3))
    L[model.free] = lam.reshape(-1, 3)
    le = X[E[:, 0]] - X[E[:, 1]]
    la = L[E[:, 0]] - L[E[:, 1]]
    return -q * np.einsum("ij,ij->i", le, la)


class Closure:
    """One closure, holding the current accepted state (x, p) and its parameter
    (p0 for prescribed pressure, K0 = p0 V0 for the gas law, V0 for the volume constraint)."""

    def __init__(self, name, x0, p0, V0):
        self.name, self.p0, self.V0, self.K0 = name, p0, V0, p0 * V0
        self.x, self.p = x0.copy(), p0
        self.newton_its, self.fwd_fail = 0, 0

    def solve(self, theta):
        """Forward solve at theta from the current accepted state (not stored)."""
        q = np.exp(theta)
        set_q(q)
        if self.name == "presc":
            ok, x, it = newton_presc(self.x, self.p0)
            p = self.p0
        elif self.name == "gas":
            ok, x, it = newton_gas(self.x, self.K0)
            p = self.K0 / model.volume(x) if ok else np.nan
        else:
            ok, x, p, it = newton_vol(self.x, self.p, self.V0)
        self.newton_its += it
        if not ok:
            self.fwd_fail += 1
        return ok, x, p, q

    def grad(self, x, p, q):
        g = model.grad_V(x)
        KT = model.tangent(x, p)
        b = dL_dx(x)
        if self.name == "presc":
            lam = spsolve(KT, b)
        elif self.name == "gas":
            V = model.volume(x)
            B = sp.bmat([[KT, g.reshape(-1, 1)], [g.reshape(1, -1), sp.csr_matrix([[-V / p]])]],
                        format="csc")
            lam = spsolve(B, np.r_[b, 0.0])[:-1]
        else:
            Jb = sp.bmat([[KT, -g.reshape(-1, 1)], [g.reshape(1, -1), None]], format="csc")
            lam = spsolve(Jb.T.tocsc(), np.r_[b, 0.0])[:-1]
        return pullback(x, lam, q)

    def stability(self, x, p):
        """Lowest eigenvalue of this closure's own tangent in the K-norm."""
        g = model.grad_V(x)
        KT = model.tangent(x, p)
        if self.name == "presc":
            return float(lowest_four(KT)[0])
        if self.name == "gas":
            return float(lowest_four(KT, p / model.volume(x), g)[0])
        return float(lowest_four(KT, 1e9 * p / model.volume(x), g)[0])


def optimise(cl):
    """L-BFGS on theta from TH0 for closure `cl`. Returns (iterations, stop reason,
    line-search failures, theta, history dict)."""
    theta = TH0.copy()
    ok, x, p, q = cl.solve(theta)
    assert ok
    cl.x, cl.p = x, p
    f = loss(x)
    g = cl.grad(x, p, q)
    g0n = np.linalg.norm(g)
    S, Y = [], []
    H = dict(f=[f], gn=[g0n], p=[p / P_STAR], V=[model.volume(x) / V_STAR], nits=[0],
             stab=[cl.stability(x, p)], stab_it=[0])
    small, reason, ls_fail = 0, "max iterations", 0
    for k in range(1, MAXIT + 1):
        # two-loop recursion
        qv = g.copy()
        al = []
        for s_, y_ in reversed(list(zip(S, Y))):
            a = (s_ @ qv) / (y_ @ s_)
            al.append(a)
            qv -= a * y_
        if S:
            qv *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s_, y_), a in zip(zip(S, Y), reversed(al)):
            qv += s_ * (a - (y_ @ qv) / (y_ @ s_))
        d = -qv
        if d @ g >= 0:
            d, S, Y = -g, [], []
        accepted = False
        for attempt in range(2):
            t = min(1.0, TRUST / np.abs(d).max())
            for _ in range(30):
                th_new = theta + t * d
                ok, xn, pn, qn = cl.solve(th_new)
                if ok:
                    fn = loss(xn)
                    if fn <= f + C1 * t * (g @ d):
                        accepted = True
                        break
                t *= 0.5
            if accepted:
                break
            ls_fail += 1
            d, S, Y = -g, [], []
        if not accepted:
            reason = "line search failed twice"
            break
        cl.x, cl.p = xn, pn
        gn_ = cl.grad(xn, pn, qn)
        s_, y_ = th_new - theta, gn_ - g
        if y_ @ s_ > 1e-16 * np.linalg.norm(y_) * np.linalg.norm(s_):
            S.append(s_); Y.append(y_)
            S, Y = S[-MEM:], Y[-MEM:]
        rel = (f - fn) / max(abs(f), 1e-300)
        theta, f, g = th_new, fn, gn_
        H["f"].append(f); H["gn"].append(np.linalg.norm(g)); H["p"].append(pn / P_STAR)
        H["V"].append(model.volume(xn) / V_STAR); H["nits"].append(cl.newton_its)
        if k % 10 == 0:
            H["stab"].append(cl.stability(xn, pn)); H["stab_it"].append(k)
        small = small + 1 if rel < 1e-10 else 0
        if small >= 10:
            reason = "objective stalled (rel. decrease < 1e-10, 10 its)"
            break
        if np.linalg.norm(g) < 1e-8 * g0n:
            reason = "gradient < 1e-8 ||g0||"
            break
    H["stab"].append(cl.stability(cl.x, cl.p)); H["stab_it"].append(k)
    set_q(Q0)
    return k, reason, ls_fail, theta, {kk: np.array(v) for kk, v in H.items()}
