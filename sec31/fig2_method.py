r"""fig2_method.py -- Fig. 2 of the paper (Sec. 2.1.2): the force-density discretisation
of the cap on ONE mesh, h/R = 1/8, uniform q.

    (a) the triangulation at p = 0 (the reference) and at its discrete pressure maximum,
        the latter shaded by the normal component of the null vector phi of K_T (Eq. 10:
        the direction in which the branch inflates);
    (b) the section y = 0 of the deformed state against the analytic hemisphere;
    (c) p R/N0 against V/V* with the analytic constant-tension branch (Eq. 2) and the
        discrete maximum p*_h;
    (d) the smallest eigenvalue lambda_min of K^-1 K_T (Eq. 14) on the same V axis: it
        crosses zero exactly where p'_h = 0, i.e. K_T is singular at the pressure maximum.

Convergence of the discrete limit point under refinement is a RESULT and lives in
Sec. 3.1 (Fig. 3e, Table 1); this figure only shows what Sec. 2.1.2 argues.
Achromatic (greys and black): blue/orange are reserved for the two closures in Sec. 3.

RUN   python fig2_method.py              (seconds; FIG2_LEVEL changes the mesh)
      python fig2_method.py --replot     (redraw from the saved npz)
OUT   results/sec31/fig2_method.{pdf,png} and fig2_method_data.npz
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fdm_limit_point.config import Config                      # noqa: E402
from fdm_limit_point import mesh as meshmod, analytic           # noqa: E402
from fdm_limit_point.continuation import sweep, locate_fold     # noqa: E402
from fdm_limit_point.model import Model                         # noqa: E402
import style as F                                               # noqa: E402  (rcParams, fonts)

import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib import cm, colors                               # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection         # noqa: E402

LEVEL = int(os.environ.get("FIG2_LEVEL", 8))        # h/R = 1/LEVEL
OUT = os.path.join(ROOT, "results", "sec31", "fig2_method")

# 3D views: true aspect, both boxes with the same z span so the two states share one scale
ELEV, AZIM = 20.0, -58.0
ZSPAN = 1.05                 # z extent of both 3D boxes, in units of R
FACE = "#ececec"             # mesh faces
EDGE = "#333333"             # mesh edges
RIM = "#000000"              # fixed rim nodes
# phi shading: Greys truncated so the darkest face still shows its edges.
PHI_CMAP = colors.LinearSegmentedColormap.from_list(
    "phi_greys", plt.get_cmap("Greys")(np.linspace(0.06, 0.62, 256)))
LINE_DISCRETE = "black"
LINE_ANALYTIC = "black"       # same as the section in (b): dash carries identity


def lam_min(model, xu, p):
    """Smallest eigenvalue of K_T phi = lambda K phi (shift-invert about -1; lambda > -1 here)."""
    from scipy.sparse.linalg import eigsh
    KT = model.tangent(xu, p)
    w = eigsh(KT, k=1, M=model.K3, sigma=-1.0, which="LM", tol=1e-10,
              return_eigenvectors=False)
    return float(w[0])


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------
def solve(cfg):
    t0 = time.time()
    msh = meshmod.build(LEVEL, cfg)
    model = Model(msh, cfg, "uniform")
    rows, states = sweep(model, cfg, log=lambda *a: None)
    fold = locate_fold(model, cfg, states, log=print)
    if fold is None:
        raise RuntimeError(f"h/R = 1/{LEVEL}: no pressure maximum on the branch")

    xyz = model.full_xyz(fold["xu"])
    # normal component of phi at every node (rim nodes are fixed: 0)
    nrm = model.nodal_normals(fold["xu"])
    phin = np.zeros(len(xyz))
    phin[model.free] = np.einsum("ij,ij->i", fold["phi1"].reshape(-1, 3),
                                 nrm[model.free])
    phin /= np.abs(phin).max()

    d = dict(
        v=np.array([r["V_over_Vstar"] for r in rows]),
        p=np.array([r["pR_over_N0"] for r in rows]),
        mu1=np.array([r["mu1"] for r in rows]) * cfg.q / cfg.N0,   # mu1 R^2/N0
        lam1=np.array([lam_min(model, s["xu"], s["p"]) for s in states]),
        lam1_fold=lam_min(model, fold["xu"], fold["p"]),
        v_fold=fold["V"] / cfg.V_star,
        p_fold=fold["p"] * cfg.R / cfg.N0,
        mu1_fold=float(fold["mu"][0]) * cfg.R ** 2 / cfg.N0,
        mu2_fold=float(fold["mu"][1]) * cfg.R ** 2 / cfg.N0,
        pdot_fold=float(fold["pdot"]),
        volwork=float(fold["volwork"]), corr=float(fold["corr"]),
        xyz=xyz, xyz_ref=model.full_xyz(model.x_flat),
        tris=np.asarray(msh["tris"]), fixed=np.asarray(msh["fixed"]),
        h=float(msh["h"]), phin=phin,
        n_nodes=msh["n_nodes"], n_free=msh["n_free"], n_edges=len(msh["edges"]))
    print(f"h/R = 1/{LEVEL}: {d['n_nodes']} nodes, {d['n_free']} free, "
          f"{len(d['tris'])} triangles, {d['n_edges']} edges   ({time.time() - t0:.1f} s)")
    print(f"discrete maximum: p*_h R/N0 = {d['p_fold']:.8f}  (p*_h/p* = "
          f"{d['p_fold'] / 2.0:.6f}),  V*_h/V* = {d['v_fold']:.8f},  p' = "
          f"{d['pdot_fold']:+.2e}")
    print(f"  mu1 R^2/N0 = {d['mu1_fold']:+.3e},  mu2 R^2/N0 = {d['mu2_fold']:.4f},  "
          f"grad V . phi / (|grad V||phi|) = {d['volwork']:.4f},  "
          f"corr(phi.n, z) = {d['corr']:+.6f}")
    print(f"  mu1 at p = 0: {d['mu1'][0]:.4f}  (Dirichlet limit j01^2 = 5.7832)")
    print(f"  lambda_min(K^-1 K_T) at the maximum = {d['lam1_fold']:+.3e}, at p = 0: "
          f"{d['lam1'][0]:.6f} (exactly 1), last state: {d['lam1'][-1]:+.4f}")
    return d


# -----------------------------------------------------------------------------
# drawing
# -----------------------------------------------------------------------------
def mesh3d(ax, X, tris, rim, cfg, caption, z0, face_vals=None):
    """One true-aspect view of the triangulation, axes off, optional per-face shading.

    matplotlib depth-sorts 3D artists as wholes, so a single rim scatter would be drawn
    entirely in front of or behind the surface. The rim is split along the view
    direction and drawn as two scatters, far half first: correct occlusion for a convex
    cap. A zoomed 3D scene overflows its axes rectangle; transparent patches and
    unclipped artists keep the silhouette intact."""
    e, a = np.radians(ELEV), np.radians(AZIM)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    depth = X[rim] @ view
    far, near = rim[depth < 0.0], rim[depth >= 0.0]

    def _rim(idx):
        ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], s=3.2, c=RIM,
                   depthshade=False, linewidths=0.0, clip_on=False)

    fc = FACE if face_vals is None else PHI_CMAP(face_vals)
    _rim(far)
    coll = Poly3DCollection(X[tris], facecolors=fc, edgecolors=EDGE,
                            linewidths=0.22)
    coll.set_zsort("average")
    coll.set_clip_on(False)
    ax.add_collection3d(coll)
    _rim(near)

    R = cfg.R
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(z0 * R, (z0 + ZSPAN) * R)
    ax.set_box_aspect((2.0, 2.0, ZSPAN), zoom=1.42)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("persp")
    ax.set_axis_off()
    ax.patch.set_visible(False)
    ax.text2D(0.5, -0.02, caption, transform=ax.transAxes, fontsize=F.FS_ANN,
              ha="center", va="top")


def section(ax, geo, cfg):
    """Meridional section of the deformed state against the analytic hemisphere.
    The A2 lattice has a row of nodes on y = 0, so the section is an exact row of
    the mesh rather than an interpolation."""
    X, h = geo["xyz"], geo["h"]
    band = np.abs(X[:, 1]) < 0.2 * h
    r, z = X[band, 0], X[band, 2]
    order = np.argsort(r)
    r, z = r[order], z[order]

    t = np.linspace(0.0, np.pi, 400)
    ax.plot(cfg.R * np.cos(t), cfg.R * np.sin(t), "--", color="black", lw=0.9,
            dashes=(4, 2.2), zorder=2, label="analytic hemisphere")
    ax.plot(r, z, "-", color="0.55", lw=0.7, zorder=3)
    ax.plot(r, z, "o", color="black", ms=2.4, mew=0.0, zorder=4,
            label="FDM nodes on $y = 0$")
    ax.plot([-cfg.R, cfg.R], [0.0, 0.0], "o", color=RIM, ms=3.6, mew=0.0,
            zorder=5)

    ax.patch.set_visible(False)      # see the note in mesh3d
    ax.set_xlabel("$r/R$", labelpad=1.5)
    ax.set_ylabel("$z/R$", labelpad=1.5)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.04, 1.16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(pad=1.5)
    ax.legend(loc="lower center", handlelength=1.7, borderaxespad=0.1,
              labelspacing=0.25, ncol=2, columnspacing=1.0)


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)


def panel_b(ax_p, ax_m, d, cfg):
    vmax = d["v"].max()
    va = np.linspace(1e-4, vmax, 600) * cfg.V_star

    # top: pressure-volume branch
    ax_p.plot(va / cfg.V_star, analytic.branch(va, cfg.N0, cfg.R) * cfg.R / cfg.N0,
              "--", color=LINE_ANALYTIC, lw=1.0, dashes=(4, 2.2), zorder=2,
              label="analytic cap")
    ax_p.plot(d["v"], d["p"], "-", color=LINE_DISCRETE, lw=1.2, zorder=3,
              label=f"FDM, $h/R = 1/{LEVEL}$")
    ax_p.plot(d["v_fold"], d["p_fold"], "o", color=LINE_DISCRETE, ms=4.2, mew=0.0,
              zorder=5, label=r"discrete maximum $p^{*}_{h}$")
    ax_p.plot([1.0], [2.0], "o", mfc="none", mec=LINE_ANALYTIC, ms=6.5, mew=0.9,
              zorder=4, label=r"analytic maximum $p^{*}$")
    ax_p.set_ylabel(r"$p\,R/N_{0}$")
    ax_p.set_ylim(0.0, 2.35)
    ax_p.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_p.legend(loc="lower right", handlelength=1.8, borderaxespad=0.3,
                labelspacing=0.3)

    # bottom: smallest eigenvalue of K^-1 K_T (dimensionless, 1 at p = 0)
    ax_m.axhline(0.0, color="0.25", lw=0.6, zorder=2)
    ax_m.plot(d["v"], d["lam1"], "-", color=LINE_DISCRETE, lw=1.2, zorder=3)
    ax_m.plot(d["v_fold"], 0.0, "o", color=LINE_DISCRETE, ms=4.2, mew=0.0, zorder=5)
    ax_m.set_ylabel(r"$\lambda_{\min}$")
    ax_m.text(0.985, 0.95,
              r"$\mathbf{K}_{T}\boldsymbol{\varphi} = \lambda\,\mathbf{K}\boldsymbol{\varphi}$",
              transform=ax_m.transAxes, fontsize=F.FS_ANN, ha="right", va="top",
              linespacing=1.3)
    ax_m.set_xlabel(r"$V/V^{*}$")
    lo = min(d["lam1"].min(), 0.0)
    ax_m.set_ylim(lo - 0.06, d["lam1"].max() + 0.06)
    ax_m.text(d["v_fold"] + 0.03, 0.06, r"$\mathbf{K}_{T}$ singular",
              fontsize=F.FS_ANN, ha="left", va="bottom")

    # the link between the two: p'_h = 0 and lambda_min = 0 at the same state
    for ax in (ax_p, ax_m):
        ax.axvline(d["v_fold"], color="0.25", lw=0.6, ls=(0, (1.2, 1.6)), zorder=1)
        ax.set_xlim(0.0, vmax)
        _spines(ax)
    ax_p.tick_params(labelbottom=False)


def make(d, cfg):
    w_in = F.WIDTH_MM / 25.4
    fig = plt.figure(figsize=(w_in, 0.46 * w_in))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.90], wspace=0.20,
                             left=0.045, right=0.985, bottom=0.095, top=0.935)
    left = outer[0].subgridspec(2, 2, height_ratios=[0.78, 1.0],
                                hspace=0.20, wspace=0.03)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.10)
    ax1 = fig.add_subplot(left[0, 0], projection="3d", computed_zorder=False)
    ax2 = fig.add_subplot(left[0, 1], projection="3d", computed_zorder=False)
    ax3 = fig.add_subplot(left[1, :])
    ax_p = fig.add_subplot(right[0])
    ax_m = fig.add_subplot(right[1], sharex=ax_p)

    face_phi = np.clip(d["phin"][d["tris"]].mean(axis=1), 0.0, 1.0)
    mesh3d(ax1, d["xyz_ref"], d["tris"], d["fixed"], cfg,
           "reference, $p = 0$", -0.5 * ZSPAN)
    mesh3d(ax2, d["xyz"], d["tris"], d["fixed"], cfg,
           r"discrete pressure maximum, shaded by $\boldsymbol{\varphi}\cdot\mathbf{n}$",
           0.0, face_vals=face_phi)
    section(ax3, dict(xyz=d["xyz"], h=d["h"]), cfg)
    panel_b(ax_p, ax_m, d, cfg)

    # colour bar for phi.n, small, tucked under the deformed cap
    cax = ax2.inset_axes([1.10, 0.14, 0.032, 0.56])   # clear of the cap's rim
    sm = cm.ScalarMappable(norm=colors.Normalize(0.0, 1.0), cmap=PHI_CMAP)
    cb = fig.colorbar(sm, cax=cax, ticks=[0.0, 0.5, 1.0])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=F.FS_TICK - 0.5, width=0.4, length=1.8, pad=1.0)
    cb.set_label(r"$\boldsymbol{\varphi}\cdot\mathbf{n}$ (norm.)",
                 fontsize=F.FS_ANN, labelpad=1.5)

    ax1.text2D(0.0, 1.0, "(a)", transform=ax1.transAxes, fontsize=F.FS_PANEL,
               fontweight="bold", ha="left", va="top")
    for ax, lab, x, y in ((ax3, "(b)", -0.16, 1.10), (ax_p, "(c)", -0.155, 1.06),
                          (ax_m, "(d)", -0.155, 1.12)):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=F.FS_PANEL,
                fontweight="bold", ha="left", va="top")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=600)
    plt.close(fig)
    for ext in (".pdf", ".png"):
        print(f"wrote {OUT + ext}  ({os.path.getsize(OUT + ext) / 1e3:.0f} kB)")


if __name__ == "__main__":
    cfg = Config()
    if "--replot" in sys.argv:          # redraw from the saved npz, no solve
        d = dict(np.load(OUT + "_data.npz"))
    else:
        d = solve(cfg)
        np.savez(OUT + "_data.npz", **{k: v for k, v in d.items()})
    make(d, cfg)
