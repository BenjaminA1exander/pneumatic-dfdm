r"""fig3_verification.py -- Fig. 3 of the paper (Sec. 3.1): a global fold and a local fold on
ONE mesh, h/R = 1/24, uniform q against zoned q.

Every panel is plotted against each cap's own limit point (V_f, p_f), so both folds sit
at (1, 1).
    (a) meridional sections at the fold (and the necked bulb of the zoned cap at 3 V_f)
    (b) p/p_f against V/V_f: the equilibrium branch, shared by every closure
    (c) lambda_min of K^-1 K_T (prescribed pressure) and K^-1 K~_T (gas law) along it
    (d) Gauss-Newton curvature of L along a uniform scaling of the force densities,
        prescribed pressure against the gas law (their ratio is (1 + S)^2, Eq. 22)
    (e) mesh convergence of p_f, z_f and the gas-law lambda_min: uniform against the
        analytic cap, zoned against the finest mesh (h/R = 1/192)
    (f) design towards a target form just past its own limit point, from starts below
        it: L/L0 against iteration; crosses mark where prescribed pressure stops
        (the four runs that stop on the limit point, h/R = 1/24; cap_optimise_target.py)

SOURCES (read only; newest file in results/sec31 or results/sec31/paper)
    pin_h24_...npz          cap_model.py (both caps)                  (b), (c)
    verify_h24_...npz       cap_verify.py (both caps)                  (d), the folds
    fold_h{6..192}_...npz   cap_verify.py FOLD_ONLY=1 (both caps)      (e)
    capopt_fold_*_h24_*.npz cap_optimise_target.py; CAPOPT_DIR         (f)
                            defaults to results/capopt_fold/paper
Sections in (a) are recomputed by volume-controlled Newton.

RUN   python fig3_verification.py          (TRI_LEVEL defaults to 24, CAP to zoned)
OUT   results/sec31/fig3_verification_h24_w0.125_blend.{png,pdf}
"""
import glob
import os
import sys

os.environ.setdefault("CAP", "zoned")
os.environ.setdefault("TRI_LEVEL", "24")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import cap_model as B                                                   # noqa: E402
from fdm_limit_point.continuation import newton as vc_newton            # noqa: E402
from style import C_MASS, C_CLASS, C_REF, C_FOLD, WIDTH_IN, FS_ANN, panel_label, tidy  # noqa: E402

if B.UNIFORM:
    raise SystemExit("run with CAP=zoned (the default): the figure needs the zoned force densities")
L = B.LEVEL
OUT = os.path.join(B.OUT, f"fig3_verification_h{L}{B.WT}")
DPI = int(os.environ.get("DPI", "200"))
CAPOPT = os.environ.get("CAPOPT_DIR", os.path.join(ROOT, "results", "capopt_fold", "paper"))


def newest(pat):
    return np.load(B.newest(pat), allow_pickle=True)


CASES = {
    "uniform": dict(pin=newest(f"pin_h{L}_g1_b1_rb0.5_2*.npz"), ver=newest(f"verify_h{L}_g1_b1_rb0.5_2*.npz"),
                    w=0.0, mk_fill=True, label="uniform $q$, global fold"),
    "zoned": dict(pin=newest(f"pin_h{L}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_2*.npz"),
                  ver=newest(f"verify_h{L}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_2*.npz"),
                  w=0.5, mk_fill=False, label="zoned $q$, local fold"),
}


def shade(c, w):
    r = np.array(to_rgb(c))
    return tuple(r + (1.0 - r) * w)


def logticks(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10.0, numticks=12))
    a.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    a.set_minor_formatter(NullFormatter())


def set_q(qv):
    Qd = sp.diags(qv)
    B.model.K = (B.Cu.T @ Qd @ B.Cu).tocsc()
    B.model.D = (B.Cu.T @ Qd @ B.Cs).tocsc()
    B.model.K3 = sp.kron(B.model.K, sp.eye(3), format="csc")
    B.model.D3 = sp.kron(B.model.D, sp.eye(3), format="csc")


def shapes_at(Vts):
    x, p, Vb, out = B.model.x_flat.copy(), 0.0, 0.0, []
    for Vt in Vts:
        while Vb < Vt - 1e-14:
            step = min(0.01 * B.VS, Vt - Vb)
            ok, x1, p1, _, _ = vc_newton(B.model, x, p, Vb + step, B.cfg)
            assert ok
            x, p, Vb = x1, p1, Vb + step
        out.append(B.model.full_xyz(x))
    return out


for k, c in CASES.items():
    c["Vf"], c["pf"] = float(c["ver"]["Vf"]), float(c["ver"]["pf"])
    R = c["pin"]["rows"]
    cols = list(c["pin"]["cols"])
    c["path"] = {n: R[:, i] for i, n in enumerate(cols)}
    keys = list(c["ver"]["keys"])
    T = c["ver"]["table"]
    c["tab"] = {n: T[:, i] for i, n in enumerate(keys)}
    c["u"] = c["tab"]["V"] / c["Vf"] - 1

# sections
q_zoned = B.q.copy()
set_q(np.full_like(B.q, B.cfg.q))
sec_uniform = shapes_at([CASES["uniform"]["Vf"]])
set_q(q_zoned)
sec_zoned = shapes_at([CASES["zoned"]["Vf"], 3.0 * CASES["zoned"]["Vf"]])

fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 0.66))
gs = fig.add_gridspec(2, 3, left=0.07, right=0.99, bottom=0.15, top=0.955, wspace=0.36, hspace=0.5)
ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
a, b, cpan, d, e, f = ax
FOLD_LINE = dict(color=C_FOLD, lw=0.7, ls=(0, (4, 2)), zorder=1.5)

# (a) sections
P0 = B.mesh["xyz"][:, :2]
sec = np.where(np.abs(P0[:, 1]) < 1e-9)[0]
sec = sec[np.argsort(P0[sec, 0])]
bn = [i for i in sec if abs(np.hypot(*P0[i]) - B.RB) <= B.W / 2]
X = sec_uniform[0]
a.plot(X[sec, 0], X[sec, 2], color="k", lw=1.3, ls="-", label=r"uniform, $V = V_f$")
X = sec_zoned[0]
a.plot(X[sec, 0], X[sec, 2], color="0.35", lw=1.3, ls=(0, (5, 2)), label=r"zoned, $V = V_f$")
a.plot(X[bn, 0], X[bn, 2], "s", ms=2.6, color="0.35")
X = sec_zoned[1]
a.plot(X[sec, 0], X[sec, 2], color="0.6", lw=1.1, ls=(0, (1, 1.2)), label=r"zoned, $V = 3V_f$")
a.plot(X[bn, 0], X[bn, 2], "s", ms=2.6, color="0.6")
a.set_aspect("equal")
a.set_xlim(-1.05, 1.05)
a.set_ylim(-0.03, 1.5)
a.set_yticks([0, 0.5, 1.0, 1.5])
a.set_xlabel(r"$x/R$")
a.set_ylabel(r"$z/R$")
Xu, Xz1, Xz3 = sec_uniform[0], sec_zoned[0], sec_zoned[1]
a.text(0.0, Xz3[sec, 2].max() + 0.05, r"zoned, $3V_f$", fontsize=FS_ANN, color="0.45", ha="center", va="bottom")
a.text(0.0, 0.5 * (Xz1[sec, 2].max() + Xu[sec, 2].max()), r"zoned, $V_f$", fontsize=FS_ANN, color="0.3",
       ha="center", va="center")
a.text(0.0, Xu[sec, 2].max() + 0.04, r"uniform, $V_f$", fontsize=FS_ANN, color="k", ha="center", va="bottom")
tidy(a)

# (b) equilibrium branch (shared by every closure), (c) lambda_min (stability under each closure)
for k, c in CASES.items():
    P = c["path"]
    V = P["V_over_Vstar"] * B.VS / c["Vf"]
    p = P["p_over_pstar"] * B.cfg.p_star / c["pf"]
    sel = V <= 3.6
    mf = (lambda col: col) if c["mk_fill"] else (lambda col: "white")
    cm, cc = shade(C_MASS, c["w"]), shade(C_CLASS, c["w"])
    ck = "0.1" if c["mk_fill"] else "0.55"
    every = max(1, int(sel.sum() / 28))
    b.plot(V[sel], p[sel], color=ck, lw=1.3, ls="-", marker="o", ms=2.6, mfc=mf(ck), mec=ck, mew=0.6,
           markevery=every, zorder=3)
    cpan.plot(V[sel], P["lam1_KTg"][sel], color=cm, lw=1.3, ls="-", marker="o", ms=2.6, mfc=mf(cm), mec=cm,
              mew=0.6, markevery=every, zorder=3)
    cpan.plot(V[sel], P["lam1_KT"][sel], color=cc, lw=1.3, ls="--", marker="s", ms=2.6, mfc=mf(cc), mec=cc,
              mew=0.6, markevery=every, zorder=4)
for axx in (b, cpan):
    axx.axvline(1.0, **FOLD_LINE)
    axx.set_xlim(0, 3.6)
    axx.set_xlabel(r"$V/V_f$")
    tidy(axx)
b.plot([1], [1], marker="o", ms=4.0, mfc="white", mec=C_FOLD, mew=1.0, ls="none", zorder=7)
b.set_ylim(0, 1.1)
b.set_ylabel(r"$p/p_f$")
b.legend(handles=[Line2D([], [], color="0.1", lw=1.3, marker="o", ms=2.6, label="uniform $q$"),
                  Line2D([], [], color="0.55", lw=1.3, marker="o", ms=2.6, mfc="white", label="zoned $q$")],
         loc="lower right")
cpan.axhline(0, color=C_REF, lw=0.6, zorder=1)
cpan.set_ylabel(r"$\lambda_{\min}$")

# (d) Gauss-Newton curvature of L along a uniform scaling of the force densities, 1^T G 1 with
#     G = (dx/dtheta)^T H_L dx/dtheta; equals the curvature along ln p (prescribed) or ln K_gas (gas law),
#     since equilibria depend only on p/q. cP, cG from the verify files; their ratio is (1 + S)^2.
for k, c in CASES.items():
    cm, cc = shade(C_MASS, c["w"]), shade(C_CLASS, c["w"])
    mf = (lambda col: col) if c["mk_fill"] else (lambda col: "white")
    Vr = c["tab"]["V"] / c["Vf"]
    o = np.argsort(Vr)
    Vr, cP, cG = Vr[o], c["tab"]["cP"][o], c["tab"]["cG"][o]
    for side in (Vr < 1.0, Vr > 1.0):                     # not joined across V_f
        d.semilogy(Vr[side], cP[side], color=cc, lw=1.2, ls="--", marker="s", ms=2.6, mfc=mf(cc), mec=cc, mew=0.6,
                   zorder=4)
    d.semilogy(Vr, cG, color=cm, lw=1.2, ls="-", marker="o", ms=2.6, mfc=mf(cm), mec=cm, mew=0.6, zorder=3)
d.axvline(1.0, **FOLD_LINE)
d.set_xlim(0.9, 1.1)
d.set_ylim(1e-2, 1e8)
d.set_xlabel(r"$V/V_f$")
d.set_ylabel(r"$\|(\mathrm{d}\mathbf{x}/\mathrm{d}\mathbf{q})\,\mathbf{q}\|^2_{\mathbf{H}_{\mathcal{L}}}$")
logticks(d)
d.set_yticks([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
tidy(d)

# (e) mesh convergence of the fold quantities. References: analytic for the uniform p_f and z_f, the h/R = 1/192
#     run otherwise (no extrapolation); points above 1e5 dof left out. Uniform lambda_min is the m = 1 sway pair
#     (volume-free, untouched by the gas law); the lattice and rim split the pair unless n_rim is a multiple of 3
#     (C3 symmetry), so the pair mean is plotted. Zoned lambda_min is the single m = 0 mode the gas law lifts off zero.
DOFMAX = 1e5
dof_u, p_u, z_u, lam_u = [], [], [], []
for Lu in (6, 12, 24, 48, 96, 192):
    fu = B.newest(f"fold_h{Lu}_g1_b1_rb0.5_2*.npz", must=False)
    if fu:
        zu = np.load(fu)
        dof_u.append(int(zu["n_dof"]))
        p_u.append(float(zu["pf"]) / B.cfg.p_star)                   # p_f / p* (analytic)
        z_u.append(float(zu["z_apex"]) / B.cfg.R)                    # z_f / R (analytic R)
        lam_u.append(float(np.mean(zu["lamG_fold"][:2])))             # m = 1 doublet mean
dof_u, p_u, z_u, lam_u = map(np.array, (dof_u, p_u, z_u, lam_u))
ZF = []
for Lz in (12, 24, 48, 96, 192):
    fz = B.newest(f"fold_h{Lz}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_2*.npz", must=False)
    if fz and "z_apex" in np.load(fz).files:
        z = np.load(fz)
        ZF.append((int(z["n_dof"]), float(z["pf"]), float(z["z_apex"]), float(np.ravel(z["lamG_fold"])[0])))
ZF = np.array(ZF)
lam_u = lam_u / lam_u[-1]                          # reference: the finest run (h/R = 1/192)
ZF[:, 1:] = ZF[:, 1:] / ZF[-1, 1:]
su, sz = dof_u <= DOFMAX, ZF[:, 0] <= DOFMAX


def nrm(y, sel):                                   # relative error |f_h/f_ref - 1| (y is already f_h / f_ref)
    return np.abs(y[sel] - 1.0)


for y, ls, mk, lab in ((p_u, "-", "o", r"$p_f$"), (z_u, ":", "^", r"$z_f$"), (lam_u, "-.", "s", r"$\tilde\lambda_{\min}$")):
    r = nrm(y, su)
    ok = r > 0
    e.loglog(dof_u[su][ok], r[ok], color="0.1", lw=1.2, ls=ls, marker=mk, ms=3.2, label=f"uniform, {lab}")
for j, (ls, mk, lab) in enumerate((("-", "o", r"$p_f$"), (":", "^", r"$z_f$"), ("-.", "s", r"$\tilde\lambda_{\min}$")), start=1):
    r = nrm(ZF[:, j], sz)
    ok = r > 0
    e.loglog(ZF[sz, 0][ok], r[ok], color="0.55", lw=1.2, ls=ls, marker=mk, ms=3.2, mfc="white", label=f"zoned, {lab}")


def slope_triangle(ax, x0, y0, decades, slope, label, below=True):
    """Small slope triangle in log-log axes: hypotenuse of the given slope (per decade of x) from (x0, y0) over
    `decades` decades of x, with a horizontal and a vertical leg; label beside the vertical leg."""
    x1, y1 = x0 * 10 ** decades, y0 * 10 ** (slope * decades)
    corner = (x1, y0) if below else (x0, y1)
    ax.fill([x0, x1, corner[0]], [y0, y1, corner[1]], facecolor="none", edgecolor=C_REF, lw=0.7, zorder=1)
    ax.text(np.sqrt(x0 * x1), min(y0, y1) / 1.5, label, fontsize=6.5, color=C_REF, ha="center", va="top")


# h ~ dof^(-1/2), so O(h) = -1/2 and O(h^2) = -1 per decade of dof
slope_triangle(e, 3.0e2, 2.5e-1, 0.5, -0.5, r"$O(h)$", below=False)
slope_triangle(e, 1.2e4, 1.6e-4, 0.5, -1.0, r"$O(h^2)$", below=False)
e.set_ylim(1e-5, 1.0)
e.set_xlabel("degrees of freedom")
e.set_ylabel(r"$|f_h/f_{\mathrm{ref}} - 1|$")
logticks(e)
tidy(e)
e.legend(handles=[Line2D([], [], color="0.2", lw=1.2, ls=ls_, marker=mk_, ms=3.2, label=lab_)
                  for ls_, mk_, lab_ in (("-", "o", r"$p_f$"), (":", "^", r"$z_f$"),
                                         ("-.", "s", r"$\tilde\lambda_{\min}$"))],
         loc="lower left", borderaxespad=0.3, handlelength=2.4, labelspacing=0.25)

# (f) optimisation to a target form 5% past its own limit point (cap_optimise_target.py), h/R = 1/24, two
#     starts below the limit point per cap: the runs that stop on the limit point (uniform 0.47 V_f, which creeps,
#     and zoned 0.99 V_f, which jumps the fold, are omitted and disclosed in the text). Uniform dark, zoned lighter.
#     Cross = prescribed pressure stops.
for fn, w in [("uniform_h24_s0.7", 0.0), ("uniform_h24_s0.9895", 0.0),
              ("zoned_blend_h24_s0.4", 0.5), ("zoned_blend_h24_s0.7", 0.5)]:
    CO = np.load(os.path.join(CAPOPT, f"capopt_fold_{fn}_eps0.05.npz"))
    fg = CO["gas_f"]
    f.semilogy(np.arange(len(fg)), fg / fg[0], color=shade(C_MASS, w), ls="-", lw=1.2, zorder=3)
    fp = CO["presc_f"]
    f.semilogy(np.arange(len(fp)), fp / fp[0], color=shade(C_CLASS, w), ls="--", lw=1.2, zorder=4)
    if len(fp) - 1 < 1000:
        f.plot([len(fp) - 1], [fp[-1] / fp[0]], ls="none", marker="x", ms=4.5, mew=1.1, color=shade(C_CLASS, w),
               zorder=5)
f.set_xlabel("iteration")
f.set_ylabel(r"$\mathcal{L}/\mathcal{L}_{0}$")
f.set_xlim(0, 1000)
f.set_ylim(1e-9, 3.0)
logticks(f)
f.set_yticks([1e0, 1e-3, 1e-6, 1e-9])
tidy(f)

for axx, lab in zip(ax, "abcdef"):
    bb = axx.get_position()
    panel_label(fig, (bb.x0 - 0.055, bb.y1 + 0.03), f"({lab})")
handles = [Line2D([], [], color=C_MASS, lw=1.3, ls="-", marker="o", ms=3, label="gas law"),
           Line2D([], [], color=C_CLASS, lw=1.3, ls="--", marker="s", ms=3, label="prescribed pressure"),
           Line2D([], [], color="0.3", ls="none", marker="o", ms=3.3, mfc="0.3", mec="0.3", label="uniform $q$ (global fold)"),
           Line2D([], [], color="0.3", ls="none", marker="o", ms=3.3, mfc="white", mec="0.3",
                  label="zoned $q$ (local fold), lighter"),
           Line2D([], [], label="fold", **{k: v for k, v in FOLD_LINE.items() if k != "zorder"})]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, handlelength=2.0,
           columnspacing=1.2)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=DPI)
print("saved", OUT, {k: (c["Vf"] / B.VS, c["pf"] / B.cfg.p_star) for k, c in CASES.items()})
