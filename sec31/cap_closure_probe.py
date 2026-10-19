r"""cap_closure_probe.py -- the three closures at the limit point of one cap (Table 1, claims iii-iv).

At the fold V_f of the cap selected by CAP / GAMMA / BETA (V_f read from the newest
fold_h<L>_... file written by cap_verify.py with FOLD_ONLY=1), with phi the null vector
of K_T (lowest eigenvector of (K_T, K)) and g = grad V, the script reports

    c_phi    = (p/V) (g^T phi)^2 / (phi^T K phi)   the rank-one term of Eq. 18 along the
                                                   critical mode, in units of the network
                                                   stiffness (claim iv); dimensionless, so
                                                   independent of scale
    lam_P    = lambda_min(K^-1 K_T)                 prescribed pressure (zero at the fold)
    lam_2    = second eigenvalue of K^-1 K_T
    lam_G    = lambda_min(K^-1 (K_T + (p/V) g g^T)) gas law (Table 1)
    lam_V    = lambda_min(K^-1 (K_T + a g g^T)), a = 1e8 p/V and 1e10 p/V
                                                   volume constraint, the limit a -> inf
                                                   of Eq. 21 (Table 1, "vol. constraint")
Interlacing requires lam_P <= lam_G <= lam_V <= lam_2; it is checked.

The state at V_f is rebuilt by volume-controlled Newton in 0.02 V* steps (the same
continuation as the sections of Fig. 3a).

RUN   CAP=uniform TRI_LEVEL=24 python cap_closure_probe.py
      CAP=zoned   TRI_LEVEL=24 python cap_closure_probe.py
OUT   results/sec31/closureprobe_h<L>_g<GAMMA>_b<BETA>[_w<W>_blend]_<time>.npz (new file per run) + stdout
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

model, cfg, VS, K3 = B.model, B.cfg, B.VS, B.K3
uniform = B.UNIFORM
pat = f"fold_h{B.LEVEL}_g1_b1_rb{B.RB:g}_2*.npz" if uniform else f"fold_h{B.LEVEL}_g{B.GAMMA:g}_b{B.BETA:g}_rb{B.RB:g}{B.WT}_2*.npz"
ff = B.newest(pat)
F = np.load(ff)
Vf = float(F["Vf"])

t0 = time.time()
x, p, V = model.x_flat.copy(), 0.0, 0.0
while V < Vf - 1e-15:
    Vt = min(V + 0.02 * VS, Vf)
    ok, x1, p1, it, st = vc_newton(model, x, p, Vt, cfg)
    assert ok, Vt
    x, p, V = x1, p1, Vt
KT, g, Vv = st["KT"], st["gV"], st["V"]
wP, vP = B.lowest(KT, k=3)
phi = vP[:, 0]
c_phi = (p / Vv) * float(g @ phi) ** 2 / float(phi @ (K3 @ phi))
wG, _ = B.lowest(KT, p / Vv, g, k=2)
wV8, _ = B.lowest(KT, 1e8 * p / Vv, g, k=2)
wV10, _ = B.lowest(KT, 1e10 * p / Vv, g, k=2)
print(f"[probe] {'uniform' if uniform else 'zoned' + B.WT}  h/R = 1/{B.LEVEL}  dof {B.N}  V_f/V* = {Vf / VS:.8f}  "
      f"p_f/p* = {p / cfg.p_star:.8f}  (fold file {os.path.basename(ff)})")
print(f"[probe] lam_P {wP[0]:+.3e}  lam_2 {wP[1]:.6f}  lam_G {wG[0]:.6f}  lam_V(1e8) {wV8[0]:.6f}  "
      f"lam_V(1e10) {wV10[0]:.6f}  c_phi {c_phi:.6f}  interlacing ok: "
      f"{wP[0] <= wG[0] + 1e-12 and wG[0] <= wV10[0] + 1e-9 and wV10[0] <= wP[1] + 1e-9}  ({time.time() - t0:.0f} s)")
print(f"[table1] vol. constraint lambda_min at V_f = {wV10[0]:.3f}   (gas law {wG[0]:.3f}; c_phi = {c_phi:.3f})")
tag = f"closureprobe_h{B.LEVEL}_g{B.GAMMA:g}_b{B.BETA:g}{'' if uniform else B.WT}_{time.strftime('%Y%m%d-%H%M%S')}"
np.savez(os.path.join(B.OUT, tag + ".npz"), level=B.LEVEL, n_dof=B.N, Vf=Vf, pf=p, lam_P=wP[0], lam_2=wP[1],
         lam_G=wG[0], lam_V8=wV8[0], lam_V10=wV10[0], c_phi=c_phi, fold_file=os.path.basename(ff))
print(f"[probe] -> {tag}.npz")
