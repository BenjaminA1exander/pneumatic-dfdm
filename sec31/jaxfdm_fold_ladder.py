r"""jaxfdm_fold_ladder.py -- the discrete limit point located with JAX-FDM alone (Sec. 3.1, claim i).

Gas-law equilibria from jax_fdm's ConstantMassModelSparse (jaxfdm_gaslaw/common.py),
damped Picard (beta = DAMP_BETA), warm-started continuation in K_gas. Only forward
equilibrium solves -- no adjoint, no dense operator, no eigenvalues -- so it scales to
h/R = 1/48. The limit point is the maximum of p along the branch:
    1. geometric ladder in K_gas until p decreases (bracket),
    2. golden section on p(K_gas) to a relative bracket of 1e-3,
    3. 9 states across the bracket; cubic fits p(V), z_apex(V); V*_h solves dp/dV = 0.
Every state passes the equilibrium gate mean||K^-1 P(x) - x|| < RES_TOL (x R).

The result is compared with the limit point of the SAME discrete cap located by the
independent SciPy volume-control code (cap_verify.py FOLD_ONLY=1; agreement is ~1e-10 in
p*, ~1e-8 in V* and z*) and with the analytic cap of Sec. 2.1.1.

RUN   TRI_LEVEL=6 python jaxfdm_fold_ladder.py                       (12 s)
      TRI_LEVEL=48 JAXFDM_COMPACT=1 OUT_SUFFIX=_compact python jaxfdm_fold_ladder.py   (7 min, ~4 GB)
OUT   results/jaxfdm/jaxfdm_fold_h<L><OUT_SUFFIX>.npz and a comparison printout
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np                                           # noqa: E402
import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

import jaxfdm_gaslaw.common as L                             # noqa: E402
from jaxfdm_gaslaw.common import (bp, structure, free_of, make_mass_model, THETA0,  # noqa: E402
                                  QDTYPE, DAMPED, APEX, V0, TMAX, N_DOF)

RD = os.path.join(ROOT, "results", "jaxfdm")
os.makedirs(RD, exist_ok=True)
LEVEL = L.TRI_LEVEL
SFX = os.environ.get("OUT_SUFFIX", "")
OUT = os.path.join(RD, f"jaxfdm_fold_h{LEVEL}{SFX}.npz")
ETA = 1e-14
RES_TOL = 1e-12
Q = jnp.exp(THETA0).astype(QDTYPE)
GR = (np.sqrt(5.0) - 1.0) / 2.0


@jax.jit
def _solve(gk, xf_init):
    """One compiled gas-law solve per mesh: gas_k is a traced argument, so the
    continuation never recompiles. Returns (free xyz, V, z_apex, fixed-point residual)."""
    mm = make_mass_model(gk, ETA)
    xf = mm.equilibrium_iterative_xyz(Q, bp.xyz_fixed, bp.loads, structure,
                                      xyz_free_init=xf_init, tmax=TMAX, eta=ETA,
                                      solver=DAMPED, implicit_diff=True)
    K = mm.stiffness_matrix(Q, structure)
    Rf = mm.residual_fixed_matrix(Q, bp.xyz_fixed, structure)
    P = mm.load_xyz_matrix_from_r_fixed((K, Rf, bp.xyz_fixed, bp.loads), xf, structure)
    res = jnp.mean(jnp.linalg.norm(mm.linearsolve_fn(K, P) - xf, axis=1))
    xyz = mm.nodes_positions(xf, bp.xyz_fixed, structure)
    V = jnp.abs(mm.enclosed_volume(xyz))
    return xf, V, xyz[APEX, 2], res


STATES = {}          # gk -> (xf, V, p, z, res)
T_SOLVE = []


def state(gk, seed=None):
    gk = float(gk)
    if gk in STATES:
        return STATES[gk]
    if seed is None:                                   # nearest computed state
        seed = STATES[min(STATES, key=lambda k: abs(np.log(k / gk)))][0]
    t = time.time()
    xf, V, z, res = _solve(jnp.asarray(gk, dtype=QDTYPE), seed)
    xf.block_until_ready()
    T_SOLVE.append(time.time() - t)
    V, z, res = float(V), abs(float(z)), float(res)
    if not (np.isfinite(V) and res < RES_TOL):
        raise RuntimeError(f"gk={gk:.6g}: equilibrium gate failed, residual {res:.2e}")
    STATES[gk] = (xf, V, gk / V, z, res)
    return STATES[gk]


def show(tag, gk):
    _, V, p, z, res = STATES[gk]
    print(f"  {tag:<7} K={gk:12.8f}  V={V:.8f}  p={p:.10f}  z_apex={z:.8f}  "
          f"res={res:.1e}  ({T_SOLVE[-1]:.1f} s)", flush=True)


t0 = time.time()
print(f"jax_fdm fold ladder, h/R = 1/{LEVEL}, {N_DOF} dof, beta = {L.DAMP_BETA}", flush=True)

# 1. bracket the maximum of p by a geometric ladder in K_gas
seed = free_of(L.vertices)
ks = []
gk = 0.25 * V0
while True:
    state(gk, seed)
    seed = STATES[gk][0]
    ks.append(gk)
    show("ladder", gk)
    if len(ks) >= 3 and STATES[ks[-1]][2] < STATES[ks[-2]][2]:
        break
    gk *= 1.25
a, c = ks[-3], ks[-1]

# 2. golden section on p(K_gas)
b1, b2 = c - GR * (c - a), a + GR * (c - a)
state(b1); state(b2)
while (c - a) / (0.5 * (a + c)) > 1e-3:
    if STATES[b1][2] > STATES[b2][2]:
        c, b2 = b2, b1
        b1 = c - GR * (c - a)
        state(b1)
    else:
        a, b1 = b1, b2
        b2 = a + GR * (c - a)
        state(b2)
show("golden", b1 if STATES[b1][2] > STATES[b2][2] else b2)

# 3. local cubic fits across the final bracket
kc = 0.5 * (a + c)
kfit = kc * (1.0 + np.linspace(-6e-3, 6e-3, 9))
for k in kfit:
    state(k)
Vf = np.array([STATES[k][1] for k in kfit])
pf = np.array([STATES[k][2] for k in kfit])
zf = np.array([STATES[k][3] for k in kfit])
Vm, Vs = Vf.mean(), Vf.std()
cp = np.polyfit((Vf - Vm) / Vs, pf, 3)
roots = np.roots(np.polyder(cp))
u = float(np.real(roots[np.argmin(np.abs(roots))]))
V_STAR = Vm + u * Vs
P_STAR = float(np.polyval(cp, u))
Z_STAR = float(np.polyval(np.polyfit((Vf - Vm) / Vs, zf, 3), u))
FIT_RES = float(np.abs(np.polyval(cp, (Vf - Vm) / Vs) - pf).max())
RES_MAX = max(s[4] for s in STATES.values())

# comparison with the independent SciPy volume-control study and the analytic cap
p_an, V_an, z_an = 2.0 * np.sqrt(3.0), 2.0 * np.pi / 3.0, 1.0   # R = 1, q = 1
F = L.scipy_fold(LEVEL)
if F is not None:
    p_sp, V_sp, z_sp = float(F["pf"]), float(F["Vf"]), float(F["z_apex"])
else:
    p_sp = V_sp = z_sp = np.nan

np.savez(OUT, level=LEVEL, n_dof=N_DOF, p_star=P_STAR, V_star=V_STAR, z_star=Z_STAR,
         gk=np.array(sorted(STATES)), V=np.array([STATES[k][1] for k in sorted(STATES)]),
         p=np.array([STATES[k][2] for k in sorted(STATES)]),
         z=np.array([STATES[k][3] for k in sorted(STATES)]),
         res=np.array([STATES[k][4] for k in sorted(STATES)]),
         fit_residual=FIT_RES, res_max=RES_MAX, eta=ETA, beta=L.DAMP_BETA,
         p_scipy=p_sp, V_scipy=V_sp, z_scipy=z_sp, seconds=time.time() - t0)

print(f"\n[fold] h/R = 1/{LEVEL}: {len(STATES)} equilibria, max residual {RES_MAX:.1e}, "
      f"cubic fit residual {FIT_RES:.1e}, {time.time() - t0:.0f} s "
      f"(median solve {np.median(T_SOLVE):.1f} s)")
print(f"{'':10}{'jax_fdm':>16}{'SciPy':>16}{'rel. diff':>11}{'err vs analytic':>17}")
for nm, j, s, an in (("p*_h", P_STAR, p_sp, p_an), ("V*_h", V_STAR, V_sp, V_an),
                     ("z*_apex", Z_STAR, z_sp, z_an)):
    print(f"{nm:10}{j:16.10f}{s:16.10f}{abs(j - s) / abs(s):11.1e}{abs(j - an) / an:17.4e}")
print(f"[saved] {OUT}")
