r"""table1.py -- assemble Table 1 of the paper from the result files of this folder.

Every entry of Table 1 is read from one of three files per cap and mesh:
    fold_<tag>.npz          cap_verify.py FOLD_ONLY=1   p_f/p*, lambda_min at V_f (prescribed, gas law)
    closureprobe_<tag>.npz  cap_closure_probe.py        lambda_min at V_f, volume constraint (a = 1e10 p/V)
    verify_<tag>.npz        cap_verify.py               gradient ratio and (1 + S)^2 at p = p_f (1 - eps),
                                                        eps = 1e-3 and 1e-7, on the stable branch
plus, as extra lines, the pole exponent n of |dL/dtheta| ~ (1 - V/V_f)^n under both closures
(verify files) and S at 6.7 V_f on the zoned cap (pin_<tag>.npz of cap_model.py, claim v).

Two tables are printed: one from the shipped reference data (results/sec31/paper, the
files behind the published table) and one from your own runs (results/sec31, newest file
per cell; a dash where no file exists).

RUN  python table1.py             (BLEND=1 is assumed for the zoned cap, as in the paper)
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fdm_limit_point.config import Config                      # noqa: E402

cfg = Config()
VS = cfg.V_star
UNI = [6, 12, 24, 48, 96, 192]
ZON = [12, 24, 48, 96, 192]
ZT = "_w0.125_blend"                                           # zone tag of the paper's zoned cap


def newest(d, pattern):
    f = sorted(glob.glob(os.path.join(d, pattern)), key=os.path.basename)
    return f[-1] if f else None


def cell(d, L, zoned):
    """All Table 1 quantities of one column, or None where the file is missing."""
    g = f"g0.25_b10_rb0.5{ZT}" if zoned else "g1_b1_rb0.5"
    out = dict(pf=None, lamP=None, lamG=None, lamV=None, ratio3=None, ratio7=None, s3=None, s7=None,
               nP=None, nG=None, S67=None)
    f = newest(d, f"fold_h{L}_{g}_2*.npz")
    if f:
        z = np.load(f)
        out["pf"] = float(z["pf"]) / cfg.p_star
        out["lamP"] = float(np.ravel(z["lamP_fold"])[0])
        out["lamG"] = float(np.ravel(z["lamG_fold"])[0])
    f = newest(d, f"closureprobe_h{L}_{'g0.25_b10' + ZT if zoned else 'g1_b1'}_2*.npz")
    if f:
        out["lamV"] = float(np.load(f)["lam_V10"])
    f = newest(d, f"verify_h{L}_{g}_2*.npz")
    if f:
        z = np.load(f, allow_pickle=True)
        keys, lab, T = list(z["keys"]), list(z["labels"]), z["table"]
        col = {k: T[:, i] for i, k in enumerate(keys)}
        for eps, tag in (("1e-03", "3"), ("1e-07", "7")):
            if f"pre {eps}" in lab:
                i = lab.index(f"pre {eps}")
                out["ratio" + tag] = col["gradP"][i] / col["gradG"][i]
                out["s" + tag] = col["onepS2"][i]
        pre = [i for i, l in enumerate(lab) if l.startswith("pre")]
        u = 1 - col["V"][pre] / float(z["Vf"])
        sel = (u > 1e-4) & (u < 1.05e-2)
        if sel.sum() >= 3:
            out["nP"] = np.polyfit(np.log(u[sel]), np.log(col["gradP"][pre][sel]), 1)[0]
            out["nG"] = np.polyfit(np.log(u[sel]), np.log(col["gradG"][pre][sel]), 1)[0]
    if zoned and out["pf"] is not None:
        f = newest(d, f"pin_h{L}_{g}_2*.npz")
        if f:
            z = np.load(f, allow_pickle=True)
            cols, R = list(z["cols"]), z["rows"]
            V = R[:, cols.index("V_over_Vstar")] * VS
            Vf = float(np.load(newest(d, f"fold_h{L}_{g}_2*.npz"))["Vf"])
            if V.max() / Vf >= 6.6:
                i = int(np.argmin(np.abs(V / Vf - 6.7)))
                out["S67"] = R[i, cols.index("S")]
    return out


def fmt(v, kind):
    if v is None:
        return "--"
    if kind == "pf":
        return f"{v:.4f}" if v > 0.9 else f"{v:.3f}"
    if kind == "e13":
        return f"{v * 1e13:.0f}" if abs(v) * 1e13 >= 2 else f"{v * 1e13:.1f}"
    if kind == "lam":
        return f"{v:.3f}"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "e6":
        return f"{v / 1e6:.2g}" if v / 1e6 < 10 else f"{v / 1e6:.0f}"
    if kind == "n":
        return f"{v:+.3f}"
    if kind == "S":
        return f"{v:.2f}"
    return str(v)


def table(d, title):
    cells = [cell(d, L, False) for L in UNI] + [cell(d, L, True) for L in ZON]
    hdr = [f"1/{L}" for L in UNI] + [f"1/{L}" for L in ZON]
    w = 7
    print(f"\n{title}")
    print(f"{'':34s}" + "uniform, global limit point".center(w * len(UNI)) + "  " + "zoned, local limit point".center(w * len(ZON)))
    print(f"{'h/R':34s}" + "".join(f"{h:>{w}s}" for h in hdr[:len(UNI)]) + "  " + "".join(f"{h:>{w}s}" for h in hdr[len(UNI):]))
    rows = [("p_f/p*", "pf", "pf"),
            ("lambda_min at V_f: prescribed, x1e-13", "lamP", "e13"),
            ("                   gas law", "lamG", "lam"),
            ("                   vol. constraint", "lamV", "lam"),
            ("gradient ratio, eps = 1e-3", "ratio3", "int"),
            ("                eps = 1e-7", "ratio7", "int"),
            ("(1 + S)^2, eps = 1e-3", "s3", "int"),
            ("           eps = 1e-7, x1e6", "s7", "e6"),
            ("pole exponent n, prescribed", "nP", "n"),
            ("                 gas law", "nG", "n"),
            ("S at 6.7 V_f", "S67", "S")]
    for name, key, kind in rows:
        vals = [fmt(c[key], kind) for c in cells]
        print(f"{name:34s}" + "".join(f"{v:>{w}s}" for v in vals[:len(UNI)]) + "  "
              + "".join(f"{v:>{w}s}" for v in vals[len(UNI):]))


if __name__ == "__main__":
    base = os.path.join(ROOT, "results", "sec31")
    table(os.path.join(base, "paper"), "Table 1 from the shipped reference data (results/sec31/paper):")
    table(base, "Table 1 from your own runs (results/sec31; dash = not run):")
