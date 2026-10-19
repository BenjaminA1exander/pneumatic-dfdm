r"""cap_bubble_check.py -- is the zoned cap's local limit point the analytic bubble of
Sec. 2.1.1 on a smaller rim? (Sec. 3.1, claim (i) for the zoned cap.)

Hypothesis: at its limit point the soft-disk bulb is a constant-tension bubble, tension
N_in = sqrt(3) q_in (Eq. 7, q_in = GAMMA q0), on the neck formed by the belt, so by
Eqs. 2-3 with R -> a
    the bulb at V_f is a hemisphere of radius a (theta* = pi/2), and  p_f = 2 N_in / a,
with a the deformed radius of the neck (inner edge of the belt). Only a is taken from the
computation; everything else is the analytic cap.

At V_f (newest fold file of cap_verify.py FOLD_ONLY=1 for this mesh) the state is rebuilt
by volume-controlled Newton (as cap_closure_probe.py) and the script reports
  a, z_neck      mean radius and height of the nodes on the inner belt edge
  r_s, c_z, dev  least-squares sphere through the soft-disk nodes: radius, centre height,
                 max node deviation
  theta          polar half-angle of that sphere's cap over the neck circle (pi/2 = hemisphere)
  p_f / p_pred   measured over predicted limit pressure, p_pred = 2 N_in / a (and 2 N_in / r_s)

RUN  CAP=zoned TRI_LEVEL=24 python cap_bubble_check.py
OUT  results/sec31/bubblecheck_h<L>_<time>.npz (new file) + one stdout line
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import cap_model as B                                                   # noqa: E402
from fdm_limit_point.continuation import newton as vc_newton            # noqa: E402

if B.UNIFORM:
    raise SystemExit("the bubble check is for the zoned cap: run with CAP=zoned")

t0 = time.time()
R, q0, VS = B.cfg.R, B.cfg.q, B.VS
ff = B.newest(f"fold_h{B.LEVEL}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_2*.npz")
F = np.load(ff)
Vf = float(F["Vf"])

x, p, V = B.model.x_flat.copy(), 0.0, 0.0
while V < Vf - 1e-15:
    Vt = min(V + 0.02 * VS, Vf)
    ok, x1, p1, _, _ = vc_newton(B.model, x, p, Vt, B.cfg)
    assert ok, Vt
    x, p, V = x1, p1, Vt

X = B.model.full_xyz(x)
P0 = B.mesh["xyz"][:, :2]
r0 = np.hypot(P0[:, 0], P0[:, 1])
r_in = B.RB - B.W / 2                                        # inner edge of the belt in the flat reference
N_in = np.sqrt(3.0) * B.GAMMA * q0
neck = np.abs(r0 - r_in) < 0.5 * B.h
inner = r0 < r_in - 1e-9
a = float(np.hypot(X[neck, 0], X[neck, 1]).mean())
a_spread = float(np.ptp(np.hypot(X[neck, 0], X[neck, 1])))
zn = float(X[neck, 2].mean())
A = np.c_[2 * X[inner], np.ones(inner.sum())]
c = np.linalg.lstsq(A, (X[inner] ** 2).sum(1), rcond=None)[0]
ctr, rs = c[:3], float(np.sqrt(c[3] + c[:3] @ c[:3]))
dev = float(np.abs(np.linalg.norm(X[inner] - ctr, axis=1) - rs).max())
theta = float(np.pi - np.arcsin(np.clip(a / rs, -1, 1))) if ctr[2] < zn else float(np.arcsin(np.clip(a / rs, -1, 1)))
p_pred = 2 * N_in / a
p_pred_s = 2 * N_in / rs
print(f"h/R = 1/{B.LEVEL}  dof {B.N}  p_f/p* {p / B.cfg.p_star:.5f}  a {a / R:.5f}R (spread {a_spread / R:.1e})  "
      f"z_neck {zn / R:.4f}R  sphere r_s {rs / R:.5f}R  c_z - z_neck {(ctr[2] - zn) / R:+.4f}R  dev {dev / R:.1e}R  "
      f"theta/(pi/2) {theta / (np.pi / 2):.4f}  p_f/(2N_in/a) {p / p_pred:.5f}  p_f/(2N_in/r_s) {p / p_pred_s:.5f}  "
      f"({time.time() - t0:.0f} s)", flush=True)
np.savez(os.path.join(B.OUT, f"bubblecheck_h{B.LEVEL}_{time.strftime('%Y%m%d-%H%M%S')}.npz"),
         level=B.LEVEL, n_dof=B.N, Vf=Vf, pf=p, p_star=B.cfg.p_star, a=a, a_spread=a_spread, z_neck=zn, r_s=rs,
         centre_z=float(ctr[2]), dev=dev, theta=theta, p_pred=p_pred, p_pred_s=p_pred_s, N_in=N_in,
         fold_file=os.path.basename(ff))
