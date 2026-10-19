r"""cap_optimise_target.py -- design under prescribed pressure and under the gas law towards
a target form JUST PAST its own limit point (Sec. 3.1, claim vi; Fig. 3f).

The test is the one prescribed pressure finds hardest while the target stays
non-degenerate:
  * every start is BELOW its limit point (stable branch of the start design);
  * the target form is an equilibrium of a known design, placed JUST PAST that design's
    own limit point:  V_t = (1 + EPS) V_f,t, with V_f,t the pressure peak of the target
    design's volume continuation (parabola through the peak).
Under prescribed pressure the start and the target lie on different branches, so every
continuous design path between them passes a singular K_T, and the target is close to
the fold, so the optimiser is drawn onto the limit point. The target is reachable by
both closures (equilibria depend on p/q only, Sec. 2.1.2), so L = 0 at finite, positive q.

Target design:  theta_t = theta_0 + TILT x/R + RAD ((r/R)^2 - mean), x and r of the flat
edge midpoints (a tilted, radially graded q).  Objective  L = n_u^-1 sum_i |x_i - x_t,i|^2.

Recorded: the history of L, |grad|, p, V and the stability check of each closure, plus
S = (p/V) g^T K_T^-1 g and the curvature ratio (1 + S)^2 (Eq. 22) at the target, at every
stability check (every 10 iterations) and at the final state of each run (for prescribed
pressure: the state where it stops).

Starts: gas-law equilibrium of the start design at K_gas = START K_f (K_f = p_f V_f of
this cap's limit point), reached by continuation from the flat state. START < 1 keeps the
start below the limit point.

RUN  TRI_LEVEL=6 CASE=uniform START=0.7 python cap_optimise_target.py        (~6 min on a laptop)
     TRI_LEVEL=24 CASE=zoned START=0.7 python cap_optimise_target.py          (the Fig. 3f runs:
                                                                               both cases, h/R = 1/24,
                                                                               START 0.4 / 0.7 / 0.9895)
ENV  TRI_LEVEL (6), CASE (uniform | zoned), START (0.4), EPS (0.05), TILT (0.4), RAD (0.6),
     CLOSURES (presc,gas; "vol" is also available), MAXIT (1000), SMOKE (0; 1 = target only), OUT_DIR
OUT  results/capopt_fold/capopt_fold_<case>_h<L>_s<START>_eps<EPS>.npz (never overwritten; the
     runs of the paper are in results/capopt_fold/paper)
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import cap_optimise_lib as A                                           # noqa: E402
from cap_optimise_lib import model, cfg, mesh, N, LEVEL, RD, newton_gas, newton_vol, lowest_four  # noqa: E402
from fdm_limit_point import zones                                      # noqa: E402

CASE = os.environ.get("CASE", "uniform")
START = float(os.environ.get("START", "0.4"))
EPS = float(os.environ.get("EPS", "0.05"))
TILT = float(os.environ.get("TILT", "0.4"))
RAD = float(os.environ.get("RAD", "0.6"))
CLOSURES = os.environ.get("CLOSURES", "presc,gas").split(",")
A.MAXIT = int(os.environ.get("MAXIT", "1000"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = os.environ.get("OUT_DIR", os.path.join(RD, "capopt_fold"))
os.makedirs(OUT, exist_ok=True)


def newest_fold(pattern):
    import glob
    d = os.path.join(RD, "sec31")
    f = sorted(glob.glob(os.path.join(d, pattern)) + glob.glob(os.path.join(d, "paper", pattern)),
               key=os.path.basename)
    assert f, f"run cap_verify.py with FOLD_ONLY=1 for this cap first ({pattern})"
    return f[-1]


E = model.edges
P0 = mesh["xyz"][:, :2]
if CASE == "zoned":
    q0, _ = zones.zone_q(P0[E[:, 0]], P0[E[:, 1]], 0.5, 0.125, 0.25, 10.0, cfg.q)
    tag = "_w0.125" + ("_blend" if zones.BLEND else "")
    F = np.load(newest_fold(f"fold_h{LEVEL}_g0.25_b10_rb0.5{tag}_2*.npz"))
    NAME = "zoned" + ("_blend" if zones.BLEND else "")
else:
    q0 = np.full(len(E), cfg.q)
    F = np.load(newest_fold(f"fold_h{LEVEL}_g1_b1_rb0.5_2*.npz"))
    NAME = "uniform"
VF, PF = float(F["Vf"]), float(F["pf"])                        # this cap's limit point (cap_verify.py)
KF = PF * VF
FILE = os.path.join(OUT, f"capopt_fold_{NAME}_h{LEVEL}_s{START:g}_eps{EPS:g}.npz")
assert not os.path.exists(FILE), f"{FILE} exists; never overwrite (set OUT_DIR)"


def S_of(xx, p):
    """S = (p/V) g^T K_T^-1 g for the design currently set in the model (Eq. 22)."""
    g = model.grad_V(xx)
    return float(p / model.volume(xx) * (g @ spsolve(model.tangent(xx, p).tocsc(), g)))


# ----------------------------------------------------------------------------- target
mid = 0.5 * (P0[E[:, 0]] + P0[E[:, 1]])
r2 = (mid ** 2).sum(1) / cfg.R ** 2
delta = TILT * mid[:, 0] / cfg.R + RAD * (r2 - r2.mean())
q_t = q0 * np.exp(delta)
A.set_q(q_t)
x, p_ = A.x_flat.copy(), 0.0
xs, Vs, ps = [], [], []
VF_t = PF_t = V_t = np.inf
k = 0
while Vs == [] or Vs[-1] < V_t:                                # volume continuation until past (1 + EPS) V_f,t
    k += 1
    assert k < 400, "no limit point found below 4 V_f"
    ok, x, p_, _ = newton_vol(x, p_, 0.01 * k * VF)
    assert ok, f"target continuation failed at V/V_f = {0.01 * k:.2f}"
    xs.append(x.copy()); Vs.append(0.01 * k * VF); ps.append(p_)
    i = int(np.argmax(ps))
    if 0 < i < len(ps) - 1 and not np.isfinite(V_t):           # first peak passed: parabola through it and neighbours
        c2, c1, c0 = np.polyfit(Vs[i - 1:i + 2], ps[i - 1:i + 2], 2)
        VF_t = -c1 / (2 * c2)
        PF_t = float(np.polyval([c2, c1, c0], VF_t))
        V_t = (1 + EPS) * VF_t
Vs, ps = np.array(Vs), np.array(ps)
j = int(np.searchsorted(Vs, V_t)) - 1                          # last stored state below V_t
ok, x_t, p_t, _ = newton_vol(xs[j], ps[j], V_t)
assert ok, "target solve failed"
V_t = model.volume(x_t)
S_t = S_of(x_t, p_t)
lam_t = float(lowest_four(model.tangent(x_t, p_t))[0])         # target design's own prescribed-pressure tangent
edge_len = lambda xx: np.linalg.norm(np.diff(model.full_xyz(xx)[E], axis=1)[:, 0], axis=1)
lmin_t = float(edge_len(x_t).min())
NU = len(x_t) // 3


def loss(xx):
    return float(np.sum((xx - x_t) ** 2) / NU)


def dL_dx(xx):
    return 2.0 * (xx - x_t) / NU


A.loss, A.dL_dx = loss, dL_dx

_stab = A.Closure.stability


def safe_stability(self, xx, p):
    """The library's stability check, plus S at the same state; never crashes a run."""
    try:
        S = S_of(xx, p)
    except Exception:
        S = np.nan
    self.S_hist.append(S)
    try:
        return _stab(self, xx, p)
    except Exception as e:                                     # ARPACK no-convergence: log, do not crash
        print(f"    [stability check failed: {type(e).__name__}]", flush=True)
        return np.nan


A.Closure.stability = safe_stability

# ----------------------------------------------------------------------------- start
A.Q0 = q0.copy()
A.TH0 = np.log(q0)
A.set_q(q0)
A.K3 = sp.kron((A.Cu.T @ sp.diags(q0) @ A.Cu).tocsc(), sp.eye(3), format="csc")   # K of the start design
A.K3_lu = splu(A.K3)
A.P_STAR, A.V_STAR = PF, VF                       # log p and V relative to this cap's own fold

x, p_ = A.x_flat.copy(), 0.0
for frac in (0.01, 0.02, 0.03, 0.04, 0.05):
    ok, x, p_, _ = newton_vol(x, p_, frac * VF)
Kg = p_ * model.volume(x)
while Kg < START * KF:
    Kg = min(Kg * 1.2, START * KF)
    ok, x, _ = newton_gas(x, Kg)
    assert ok, f"start continuation failed at K/K_f = {Kg / KF:.4f}"
V0 = model.volume(x)
p0 = Kg / V0
S0 = S_of(x, p0)
assert V0 < VF, "start is not below its limit point"

t0 = time.time()
print(f"cap optimisation to a target just past its limit point [{NAME}], h/R = 1/{LEVEL}, {N} dof, {len(E)} design "
      f"variables; fold V_f {VF:.6f} p_f {PF:.6f}", flush=True)
print(f" target: delta = {TILT:g} x/R + {RAD:g} ((r/R)^2 - mean); target design's own limit point "
      f"V_f,t/V_f = {VF_t / VF:.4f}, p_f,t/p_f = {PF_t / PF:.4f}; target V_t/V_f,t = {V_t / VF_t:.4f} "
      f"(V_t/V_f = {V_t / VF:.4f}, p_t/p_f = {p_t / PF:.4f}); its prescribed-pressure tangent lam1 {lam_t:+.4f}, "
      f"S {S_t:+.3e}, (1+S)^2 {(1 + S_t) ** 2:.3e}", flush=True)
print(f" start s = {START}: V/V_f = {V0 / VF:.4f}, p/p_f = {p0 / PF:.4f}, S {S0:+.3e}, (1+S)^2 {(1 + S0) ** 2:.3e}, "
      f"L0 = {loss(x):.6e}", flush=True)
out = dict(level=LEVEL, case=NAME, start=START, Vf=VF, pf=PF, EPS=EPS, TILT=TILT, RAD=RAD, delta=delta, q0=q0,
           x_target=x_t, p_target=p_t, V_target=V_t, Vf_target=VF_t, pf_target=PF_t, S_target=S_t,
           lam_target=lam_t, target_path_V=Vs, target_path_p=ps, x_start=x.copy(), p_start=p0, V_start=V0,
           S_start=S0)
if SMOKE:
    out["SMOKE"] = True
    np.savez(FILE.replace(".npz", "_SMOKE.npz"), **out)
    print("[smoke] target and start only, no optimisation", flush=True)
    raise SystemExit
for name in CLOSURES:
    t = time.time()
    cl = A.Closure(name, x, p0, V0)
    cl.S_hist = []
    k, reason, lsf, th, H = A.optimise(cl)
    unstable = int((H["stab"] < 0).sum())
    xf = cl.x.copy()
    pf_ = p0 if name == "presc" else (p0 * V0 / model.volume(xf) if name == "gas" else cl.p)
    A.set_q(np.exp(th))                                        # the final design, for S at the final state
    S_end = S_of(xf, pf_)
    q_tp = q_t * (pf_ / p_t)                                   # the target design scaled to this closure's pressure
    qf = np.exp(th)
    dmax = float(np.linalg.norm((xf - x_t).reshape(-1, 3), axis=1).max())
    lmin = float(edge_len(xf).min()) / lmin_t
    print(f"  {name:5s}: {k:4d} its ({reason}); L {H['f'][0]:.4e} -> {H['f'][-1]:.4e}; "
          f"|g| {H['gn'][0]:.2e} -> {H['gn'][-1]:.2e}; Newton {cl.newton_its}; fwd fail {cl.fwd_fail}; "
          f"LS fail {lsf}; end p/p_f {H['p'][-1]:.4f} V/V_f {H['V'][-1]:.4f}; lam1 own tangent "
          f"{H['stab'][0]:+.3f} -> {H['stab'][-1]:+.3e} (negative at {unstable}/{len(H['stab'])} checks); "
          f"end S {S_end:+.3e}, (1+S)^2 {(1 + S_end) ** 2:.3e}; "
          f"max node dist to target {dmax / cfg.R:.2e}R; shortest edge / target's {lmin:.3f}; "
          f"q/q_t' in [{(qf / q_tp).min():.3f}, {(qf / q_tp).max():.3f}] ({time.time() - t:.0f} s)", flush=True)
    for kk, v in H.items():
        out[f"{name}_{kk}"] = v
    out[f"{name}_meta"] = np.array([k, cl.newton_its, cl.fwd_fail, lsf])
    out[f"{name}_theta"] = th
    out[f"{name}_x"] = xf
    out[f"{name}_dmax"] = dmax
    out[f"{name}_lmin_rel"] = lmin
    out[f"{name}_S"] = np.array(cl.S_hist)                    # at the stability checks (same its as {name}_stab_it)
    out[f"{name}_S_end"] = S_end
    np.savez(FILE, **out)
print(f"[saved] {os.path.basename(FILE)} ({time.time() - t0:.0f} s)", flush=True)
