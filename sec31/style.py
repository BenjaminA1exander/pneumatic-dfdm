r"""style.py -- shared journal figure style for the Sec. 3.1 figures.

190 mm double-column width, Type-42 fonts, the blue/orange pair of the paper
(blue = gas law, orange = prescribed pressure), recessive grid and axes.

Identity is never carried by colour alone: every series also gets its own dash
pattern and marker, so the figures survive greyscale printing and colour-vision
deficiency without a separate monochrome version.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette. C_MASS is the gas-law (constant-mass) closure, C_CLASS prescribed pressure.
C_MASS = "#2456c4"      # blue   -- gas law
C_CLASS = "#d97706"     # orange -- prescribed pressure
C_REF = "#6b7280"       # grey   -- reference lines, slope guides
C_FOLD = "#111827"      # near-black -- the limit point itself

# Secondary encoding, so the pair is separable without colour.
STY_MASS = dict(color=C_MASS, lw=1.4, ls="-", marker="o", ms=3.0, mew=0.0)
STY_CLASS = dict(color=C_CLASS, lw=1.4, ls="--", marker="s", ms=3.0, mew=0.0)

FS_LABEL, FS_TICK, FS_ANN, FS_LEG, FS_PANEL = 8.0, 7.0, 6.8, 6.8, 9.0
WIDTH_MM = float(os.environ.get("FIG_W_MM", "190"))
WIDTH_IN = WIDTH_MM / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEG,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "grid.color": "0.4",
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.15,
    "legend.labelspacing": 0.35,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def panel_label(fig, xy, s):
    fig.text(xy[0], xy[1], s, fontsize=FS_PANEL, fontweight="bold",
             va="bottom", ha="left")


def tidy(ax, grid=True):
    """Recessive axes: no top/right spine, grid under the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)
    return ax


def mark_fold(ax, x=1.0, label=None, ytext=0.96, ha="left", dx=0.012):
    """Vertical rule at the limit point, plus an optional inline label."""
    ax.axvline(x, color=C_FOLD, lw=0.7, ls=(0, (4, 2)), zorder=1.5)
    if label:
        ax.text(x + dx, ytext, label, transform=ax.get_xaxis_transform(),
                fontsize=FS_ANN, color=C_FOLD, ha=ha, va="top", rotation=0)
    return ax


def save(fig, out, quiet=False):
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600)
        if not quiet:
            print(f"saved {out}.{ext}  "
                  f"({os.path.getsize(f'{out}.{ext}') / 1e6:.2f} MB)")
    plt.close(fig)
