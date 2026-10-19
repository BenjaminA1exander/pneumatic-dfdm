r"""compare_to_paper.py -- compare your re-runs with the shipped files of the paper, cell by cell.

For every result file in results/sec31 and results/jaxfdm (your runs) the newest shipped
file of the same kind, cap and mesh in the paper/ folder is found, and every numeric
array the two have in common is compared: the largest relative difference
max |a - b| / max(|b|, floor) is printed per array (floor 1e-300, so near-zero
quantities such as lambda_min at the fold, ~1e-13, compare in absolute terms through the
scale of the reference; those are listed separately with their absolute difference).
Arrays of unequal shape (different number of path states) are reported as such.

RUN  python compare_to_paper.py            (prints one block per file pair)
"""
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = [os.path.join(ROOT, "results", "sec31"), os.path.join(ROOT, "results", "jaxfdm")]
STAMP = re.compile(r"_\d{8}-\d{6}")
# key arrays whose values are quoted in the paper, compared with a tolerance; everything else is informational
TOL = 1e-6


def kind_of(name):
    """File name without its time stamp: the pattern that identifies cap, mesh and script."""
    return STAMP.sub("", os.path.basename(name))


def compare(a, b):
    out = []
    for k in sorted(set(a.files) & set(b.files)):
        if k in ("seconds", "wall_s", "newton_its"):          # wall time, machine-dependent
            continue
        x, y = a[k], b[k]
        if x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
            continue
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape:
            out.append((k, f"shape {x.shape} vs {y.shape}"))
            continue
        if x.size == 0:
            continue
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            continue
        scale = np.abs(y[m]).max()
        diff = np.abs(x[m] - y[m]).max()
        if scale < 1e-9:
            out.append((k, f"abs {diff:.1e} (reference ~{scale:.1e})"))
        else:
            out.append((k, diff / scale))
    return out


def main():
    worst = 0.0
    for d in DIRS:
        paper = {}
        for f in glob.glob(os.path.join(d, "paper", "*.npz")):
            paper.setdefault(kind_of(f), []).append(f)
        mine = {}
        for f in glob.glob(os.path.join(d, "*.npz")):
            mine.setdefault(kind_of(f), []).append(f)
        for kind in sorted(mine):
            if kind not in paper:
                print(f"\n{kind}: no shipped counterpart")
                continue
            fm, fp = sorted(mine[kind])[-1], sorted(paper[kind])[-1]
            a, b = np.load(fm, allow_pickle=True), np.load(fp, allow_pickle=True)
            res = compare(a, b)
            nums = [v for _, v in res if isinstance(v, float)]
            w = max(nums) if nums else float("nan")
            worst = max(worst, w) if nums else worst
            print(f"\n{kind}:  {os.path.basename(fm)}  vs  paper/{os.path.basename(fp)}"
                  f"\n  largest relative difference over {len(nums)} arrays: {w:.1e}")
            for k, v in res:
                if not isinstance(v, float) or v > TOL:
                    print(f"    {k:24s} {v if isinstance(v, str) else f'{v:.1e}'}")
    print(f"\nworst relative difference over all compared arrays: {worst:.1e}")


if __name__ == "__main__":
    main()
