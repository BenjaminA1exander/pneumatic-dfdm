r"""cap_model.py -- the two caps of Sec. 3.1 (uniform: global limit point; zoned: local
limit point) on one mesh, and the volume-controlled path with the spectra of Fig. 3(b,c).

Every other script in this folder imports this module for the model (`model`), the
configuration (`cfg`), the zoned force densities (`q`), the eigen-solver `lowest` and
the file helper `newest`. Run as a script it traces the equilibrium branch by
volume control and records, at every state, the quantities of Fig. 3(b) and (c).

MODEL   the pinned cap of fdm_limit_point (rim r = R fixed, flat reference, exact K_T),
        with force densities by zone of the flat reference (fdm_limit_point.zones):
            soft disk   r < RB - W/2        q = GAMMA q0
            belt        |r - RB| <= W/2     q = BETA  q0
            outer ring  r > RB + W/2        q = q0
        GAMMA = BETA = 1 is the uniform cap of Sec. 2.1.2 (one global limit point at
        the hemisphere). GAMMA = 0.25, BETA = 10, RB = 0.5, W = 0.125 is the zoned cap:
        the soft disk bulges into a bulb that necks at the belt and meets a limit point
        of its own at about 0.22 V* (Fig. 3a), the "local fold" of the paper.

PATH    volume-controlled continuation in Vbar (bordered Newton, Eq. 10, which passes
        every pressure fold), adaptive step DV.

LOGGED  at every converged state: p, V, S = (p/V) g^T K_T^-1 g (Eq. 22), the lowest NEIG
        eigenpairs of (K_T, K) and of (K~_T, K) with K~_T = K_T + (p/V) g g^T (Eq. 18),
        the number of negative eigenvalues of each, the volume content of every low mode
        split by zone (g_zone . phi), and the neck geometry (bulb radius / belt radius).
        With one negative K_T mode, det K~_T = det K_T (1 + S) predicts neg(K~_T) = 0
        iff S < -1 ("pred" column).

RUN     CAP=uniform TRI_LEVEL=24 python cap_model.py                 (Fig. 3b,c, uniform)
        CAP=zoned   TRI_LEVEL=24 V_END=1.56 python cap_model.py      (Fig. 3b,c, zoned; the
                                                                      path to 7 V_f gives S at
                                                                      6.7 V_f, claim (v))
ENV     CAP        uniform (default) | zoned: presets GAMMA, BETA, RB, BELT_W
        TRI_LEVEL  h/R = 1/TRI_LEVEL (default 12)
        GAMMA, BETA, RB, BELT_W   override the presets
        BLEND      1 (default): length-weighted zone blend on crossing edges; 0: midpoint
        V_END      end of the path in units of V* (default 3.0), DV first step (0.02)
        NEIG       eigenpairs per state (default 6)
OUT     results/sec31/pin_h<L>_g<GAMMA>_b<BETA>_rb<RB>[_w<W>_blend]_<time>.npz (new file per run)
"""
import glob
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve, eigsh, LinearOperator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fdm_limit_point.config import Config                      # noqa: E402
from fdm_limit_point import mesh as fmesh                      # noqa: E402
from fdm_limit_point.model import Model, _GRAD                 # noqa: E402
from fdm_limit_point.continuation import newton as vc_newton   # noqa: E402
from fdm_limit_point import zones                              # noqa: E402

# ----------------------------------------------------------------------------- configuration
CAP = os.environ.get("CAP", "uniform").lower()
if CAP not in ("uniform", "zoned"):
    raise SystemExit("CAP must be 'uniform' or 'zoned'")
_PRESET = dict(uniform=("1", "1"), zoned=("0.25", "10"))[CAP]
LEVEL = int(os.environ.get("TRI_LEVEL", "12"))
GAMMA = float(os.environ.get("GAMMA", _PRESET[0]))
BETA = float(os.environ.get("BETA", _PRESET[1]))
RB = float(os.environ.get("RB", "0.5"))
W = float(os.environ.get("BELT_W", "0.125"))           # belt width, units of R
V_END = float(os.environ.get("V_END", "3.0"))          # in units of V*
DV0 = float(os.environ.get("DV", "0.02"))              # in units of V*
NEIG = int(os.environ.get("NEIG", "6"))
OUT = os.path.join(ROOT, "results", "sec31")
os.makedirs(OUT, exist_ok=True)

UNIFORM = (GAMMA == 1.0 and BETA == 1.0)
# file-name tag of the zone scheme; empty for the uniform cap (its files carry g1_b1)
WT = "" if UNIFORM else f"_w{W:g}" + ("_blend" if zones.BLEND else "")
TAG = f"pin_h{LEVEL}_g{GAMMA:g}_b{BETA:g}_rb{RB:g}{WT}_{time.strftime('%Y%m%d-%H%M%S')}"


def newest(pattern, must=True):
    """Newest file matching `pattern` in results/sec31 or results/sec31/paper (the
    shipped reference data). File names end in a time stamp, so the newest sorts last."""
    files = glob.glob(os.path.join(OUT, pattern)) + glob.glob(os.path.join(OUT, "paper", pattern))
    files = sorted(files, key=os.path.basename)
    if not files:
        if must:
            raise FileNotFoundError(f"no file matches {pattern} in {OUT} (or its paper/ folder)")
        return None
    return files[-1]


# ----------------------------------------------------------------------------- model
cfg = Config()
cfg.newton_max = 30
mesh = fmesh.build(LEVEL, cfg)
model = Model(mesh, cfg, "uniform")
model.reference()
h = mesh["h"]
VS = cfg.V_star

# zones and force densities of the flat reference, then K, D rebuilt for this q
P0 = mesh["xyz"][:, :2]
E = model.edges
q, zone_e = zones.zone_q(P0[E[:, 0]], P0[E[:, 1]], RB, W, GAMMA, BETA, cfg.q)
m = len(E)
Cm = sp.coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[E[:, 0], E[:, 1]])),
                   shape=(m, mesh["n_nodes"])).tocsr()
Cu, Cs = Cm[:, model.free], Cm[:, model.fixed]
Qd = sp.diags(q)
model.K = (Cu.T @ Qd @ Cu).tocsc()
model.D = (Cu.T @ Qd @ Cs).tocsc()
model.K3 = sp.kron(model.K, sp.eye(3), format="csc")
model.D3 = sp.kron(model.D, sp.eye(3), format="csc")
K3 = model.K3
N = model.ndof

T = model.tris
r_tri = np.linalg.norm(P0[T].mean(axis=1), axis=1)
tri_in = r_tri < RB                                     # bulb side of the belt centre line
node_r = np.linalg.norm(P0, axis=1)
bulb_nodes = np.where(node_r < RB - W / 2)[0]
belt_nodes = np.where(np.abs(node_r - RB) <= W / 2)[0]


def zone_grad(xu, mask):
    """grad V restricted to the triangles in `mask` (the volume carried by one zone)."""
    g_tri = np.array(_GRAD(model._tri_coords(model.full_xyz(xu))))
    g_tri[~mask] = 0.0
    g = np.zeros(3 * mesh["n_nodes"])
    np.add.at(g, model._tri_dofs.ravel(), g_tri.ravel())
    return g[model._free_dofs]


def lowest(A, c=0.0, g=None, k=NEIG):
    """k eigenpairs of (A + c g g^T, K3) nearest -1, i.e. the lowest ones (all
    eigenvalues of K^-1 K_T exceed -1 on these caps): shift-invert about -1 with a
    sparse LU of A + K3; the rank-one term is applied by Sherman-Morrison, never
    formed. c = 0 gives the prescribed-pressure tangent, c = p/V the gas law (Eq. 18),
    c -> infinity the volume constraint."""
    lu = splu((A + K3).tocsc())
    if c == 0.0:
        opinv = LinearOperator((N, N), matvec=lu.solve)
        Aop = A
    else:
        y = lu.solve(g)
        den = 1.0 + c * (g @ y)
        opinv = LinearOperator((N, N), matvec=lambda v: (lambda z: z - y * (c * (g @ z)) / den)(lu.solve(v)))
        Aop = LinearOperator((N, N), matvec=lambda v: A @ v + c * g * (g @ v))
    w, V = eigsh(Aop, k=k, M=K3, sigma=-1.0, which="LM", OPinv=opinv, tol=1e-10)
    i = np.argsort(w)
    return w[i], V[:, i]


def neck(xu):
    """(bulb max radius / belt mean radius, bulb height)."""
    X = model.full_xyz(xu)
    rb = np.linalg.norm(X[bulb_nodes, :2], axis=1).max() if len(bulb_nodes) else np.nan
    rn = np.linalg.norm(X[belt_nodes, :2], axis=1).mean()
    zb = X[bulb_nodes, 2].max() if len(bulb_nodes) else np.nan
    return rb / rn, zb


def main():
    print(f"[pin] {CAP} cap  h/R = 1/{LEVEL}  dof {N}  GAMMA {GAMMA}  BETA {BETA}  RB {RB}  belt width {W:.3f}  "
          f"edges inner/belt/outer {np.bincount(zone_e, minlength=3)}  tag {TAG}", flush=True)
    print(f"{'V/V*':>7} {'p/p*':>8} {'S':>10} {'negKT':>5} {'negKTg':>6} {'pred':>4} {'lam1 KT':>10} {'lam2 KT':>10} "
          f"{'lam1 KTg':>10}  {'mode1 dV in/out':>17}  {'mode2 dV in/out':>17}  {'bulb r/neck':>11} {'z bulb':>7}", flush=True)

    rows, modes_KT, modes_KTg = [], [], []
    xu, p = model.x_flat.copy(), 0.0
    Vbar, dV = 0.0, DV0 * VS
    t0 = time.time()
    while Vbar < V_END * VS:
        trial = Vbar + dV
        ok, x1, p1, it, st = vc_newton(model, xu, p, trial, cfg)
        if not ok:
            dV *= 0.5
            if dV < 1e-6 * VS:
                print(f"[pin] continuation stopped at V/V* = {Vbar / VS:.4f}: step below 1e-6 V*", flush=True)
                break
            continue
        xu, p, Vbar = x1, p1, trial
        if it <= 3:
            dV = min(1.5 * dV, DV0 * VS)
        g, KT, V = st["gV"], st["KT"], st["V"]
        S = (p / V) * float(g @ spsolve(KT, g))
        wP, vP = lowest(KT)
        wG, vG = lowest(KT, p / V, g)
        nP, nG = int((wP < 0).sum()), int((wG < 0).sum())
        g_in = zone_grad(xu, tri_in)
        g_out = g - g_in
        vin = vP.T @ g_in
        vout = vP.T @ g_out
        pred = "" if nP != 1 else ("0" if S < -1 else "1")
        nr, zb = neck(xu)
        rows.append([V / VS, p / cfg.p_star, S, nP, nG, wP[0], wP[1], wG[0], nr, zb, it,
                     vin[0], vout[0], vin[1], vout[1], int(nP == NEIG), int(nG == NEIG)])
        modes_KT.append(np.c_[wP, vin, vout])
        modes_KTg.append(np.c_[wG, vG.T @ g_in, vG.T @ g_out])
        print(f"{V / VS:7.3f} {p / cfg.p_star:8.4f} {S:10.3f} {nP:5d} {nG:6d} {pred:>4} {wP[0]:10.4f} {wP[1]:10.4f} "
              f"{wG[0]:10.4f}  {vin[0]:+8.3f}/{vout[0]:+8.3f}  {vin[1]:+8.3f}/{vout[1]:+8.3f}  {nr:11.3f} {zb:7.3f}",
              flush=True)

    rows = np.array(rows)
    np.savez(os.path.join(OUT, TAG + ".npz"),
             cols=np.array(["V_over_Vstar", "p_over_pstar", "S", "neg_KT", "neg_KTg", "lam1_KT", "lam2_KT", "lam1_KTg",
                            "bulb_r_over_neck_r", "z_bulb", "newton_its", "m1_dV_in", "m1_dV_out", "m2_dV_in",
                            "m2_dV_out", "KT_count_incomplete", "KTg_count_incomplete"]),
             rows=rows, modes_KT=np.array(modes_KT, dtype=object), modes_KTg=np.array(modes_KTg, dtype=object),
             x_final=xu, p_final=p, q=q, zone_e=zone_e, level=LEVEL, gamma=GAMMA, beta=BETA, rb=RB, belt_w=W,
             wall_s=time.time() - t0)
    print(f"[pin] {len(rows)} states, max Newton iterations per step {int(rows[:, 10].max())}, "
          f"{time.time() - t0:.0f} s -> {os.path.join(OUT, TAG + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
