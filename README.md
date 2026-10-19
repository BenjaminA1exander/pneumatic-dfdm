# pneumatic-dfdm

Companion code for

> B. A. O'Regan, *Differentiable design of pneumatic membranes*, submitted to
> Computer Methods in Applied Mechanics and Engineering (2026).

The paper extends the differentiable force density method (DFDM, Pastrana et al.) from
membranes spanning a footprint to vessels enclosed on themselves. Under prescribed
pressure the tangent stiffness is singular at the pressure limit point and the DFDM
gradient has a pole there (Sec. 2.1-2.2). Closing each chamber with an isothermal gas
law, `p = K_gas / V(x)`, adds one rank-one term to the tangent (Eq. 18) that removes the
singularity, keeps the load explicit and leaves the DFDM adjoint unchanged (Sec. 2.2.1).

This repository contains the method (Sec. 2) and its verification on the pneumatic cap
(Sec. 3.1: Table 1, Fig. 2, Fig. 3). Every number of Table 1 can be regenerated here; the
coarse meshes take minutes on a laptop. The application examples of Sec. 3.2-3.4
(astigmatic reflector, drop-stitch and blended-wing-body tanks, paraboloid with VaReS
patterning and Abaqus verification) are **not** included: they are multi-day runs with
their own mesh generators and a commercial FE code, and are documented in the paper.

## What is where

```
fdm_limit_point/        reference implementation of the cap (Sec. 2.1.2), NumPy/SciPy
    config.py           every parameter (R = 1, q = 1, so N0 = sqrt(3), p* = 2 sqrt(3), V* = 2 pi / 3)
    mesh.py             equilateral (A2) triangulation of the disk, rim nodes on the circle
    analytic.py         the constant-tension spherical cap, Eqs. 1-3
    model.py            K, D, V, grad V, grad^2 V, residual (Eq. 4) and tangent K_T (Eq. 6)
    continuation.py     volume-controlled Newton (the bordered system of Eq. 10), spectra, fold location
    zones.py            the zoned force densities of the local-limit-point cap (Sec. 3.1)
jaxfdm_gaslaw/          the gas-law closure inside JAX-FDM (Sec. 2.2)
    patch/              three patched files of jax_fdm 0.10.0 (ConstantMassModelSparse = Eq. 16),
                        the diff, checksums and the installer apply_patch.py
    common.py           the cap as a JAX-FDM problem: damped Picard forward solve, adjoint rules
                        (GMRES at 1e-12 instead of the shipped NormalCG at 1e-6), speed-ups
sec31/                  the scripts of Sec. 3.1, one per quantity (see the map below)
tests/                  independent checks of fdm_limit_point (finite differences, dense eigensolvers,
                        the analytic cap, the published fold at h/R = 1/6)
results/*/paper/        the result files behind Table 1 and Fig. 3 as submitted (docs/PROVENANCE.md)
run_sec31.sh            the SciPy pipeline: Table 1, Fig. 2, Fig. 3 at the meshes you choose
run_jaxfdm.sh           the JAX-FDM pipeline: fold, spectra, pole exponent, adjoint checks
run_capopt.sh           the design runs of claim (vi) / Fig. 3(f)
```

## Installation

Python 3.10 with the pinned versions of the authors' machine (`environment.yml` or
`requirements.txt`): numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, jax/jaxlib 0.4.30,
equinox 0.11.10, lineax 0.0.7, jaxopt 0.8.5, compas 1.17.10, and jax_fdm 0.10.0 from PyPI
with the three patched files of `jaxfdm_gaslaw/patch` copied in:

```
conda env create -f environment.yml
conda activate pneumatic-dfdm
pip install jax_fdm==0.10.0 --no-deps
python jaxfdm_gaslaw/patch/apply_patch.py          # keeps the originals as *.orig; --check / --restore
python tests/test_fdm_limit_point.py               # 9 checks, ~30 s, no pytest needed
```

The SciPy side (`fdm_limit_point`, `cap_*.py`, both figures) needs only numpy, scipy,
matplotlib and jax (for the exact derivatives of the volume). The `jaxfdm_*.py` scripts
need the patched jax_fdm. Everything runs on the CPU in double precision
(`JAX_ENABLE_X64` is set by the modules).

## Reproducing Sec. 3.1

All scripts are configured by environment variables (table below) and write a new,
time-stamped file into `results/`; nothing is overwritten. Run them from the repository
root; on Windows use Git Bash for the `run_*.sh` drivers.

```
PYTHON=python LEVELS="6 12 24" ./run_sec31.sh      # ~25 min: Table 1 columns 1/6-1/24, Fig. 2, Fig. 3
PYTHON=python LEVELS="6 12"    ./run_jaxfdm.sh     # ~30 min: the JAX-FDM checks at 1/6 and 1/12
PYTHON=python LEVEL=6 CASES=uniform ./run_capopt.sh   # ~20 min: claim (vi) at 1/6
python sec31/table1.py                              # Table 1 from the shipped data and from your runs
```

| script | what it computes | paper |
|---|---|---|
| `cap_model.py` | the uniform or zoned cap; volume-controlled path with p, S, the lowest eigenvalues of (K_T, K) and (K~_T, K) at every state | Fig. 3(b,c); S at 6.7 V_f |
| `cap_verify.py` | locates the limit point (`FOLD_ONLY=1`: p_f, z_f, lambda_min); prescribed-pressure sweep to p_f(1 - eps); S by three routes; responses dx/dln p and dx/dln K_gas against finite differences; the identity of Eq. 22; Gauss-Newton curvatures; gradient norms under both closures; gas-law continuation through the fold; pole exponents; adjoint vs FD | Table 1, Fig. 3(d,e), claims (i)-(v) |
| `cap_closure_probe.py` | lambda_min of K^-1 K_T, K^-1 K~_T and the volume-constraint limit at the fold; c_phi, the rank-one term along the critical mode | Table 1 "vol. constraint", claim (iv) |
| `cap_bubble_check.py` | the zoned cap's limit point against the analytic bubble on the neck | Sec. 3.1, claim (i), zoned |
| `cap_optimise_target.py` (+ `cap_optimise_lib.py`) | L-BFGS design under each closure towards a target just past its own limit point | Fig. 3(f), claim (vi) |
| `table1.py` | assembles Table 1 from the result files | Table 1 |
| `fig2_method.py`, `fig3_verification.py` | the figures | Fig. 2, Fig. 3 |
| `jaxfdm_fold_ladder.py` | the limit point from JAX-FDM forward solves alone (max of p along K_gas) | claim (i), cross-code check |
| `jaxfdm_fold_spectra.py` | lambda_min of both adjoint operators at the fold, matrix-free from jax_fdm's load | Table 1 cross-check |
| `jaxfdm_pole_exponent.py` | gradient-norm exponent n from the jax_fdm adjoint (gmres_sparse rule) | claim (ii) |
| `jaxfdm_adjoint_reference.py` | both jax_fdm adjoints against a dense / Sherman-Morrison direct solve | "1e-12 away from the limit point, 1e-9 next to it" |
| `jaxfdm_fd_through_fold.py` | gas-law gradient against finite differences of the nonlinear solve through the fold | "1e-11 uniform, 7e-10 zoned" |

### Environment variables

| variable | default | meaning |
|---|---|---|
| `CAP` | `uniform` | `uniform` (global limit point) or `zoned` (local limit point: GAMMA 0.25, BETA 10, RB 0.5, BELT_W 0.125) |
| `TRI_LEVEL` | 12 (JAX scripts: 6) | mesh: h/R = 1/TRI_LEVEL; Table 1 uses 6, 12, 24, 48, 96, 192 |
| `GAMMA`, `BETA`, `RB`, `BELT_W` | preset by `CAP` | soft-disk and belt force-density factors, belt radius and width (units of R) |
| `BLEND` | 1 | zone assignment of crossing edges: 1 length-weighted blend (the paper), 0 midpoint |
| `FOLD_ONLY` | 0 | `cap_verify.py`: stop after locating the limit point (used for 1/96 and 1/192) |
| `V_END`, `DV` | 3.0, 0.02 | `cap_model.py`: end and first step of the path, in units of V* (zoned path to 7 V_f: 1.56) |
| `CASE`, `START`, `EPS` | uniform, 0.4, 0.05 | `cap_optimise_target.py`: cap, start K_gas/K_f, target V_t = (1 + EPS) V_f,t |
| `ZONED` | 0 | `jaxfdm_adjoint_reference.py`, `jaxfdm_fd_through_fold.py`: the zoned cap |
| `JAXFDM_COMPACT`, `OUT_SUFFIX` | 0, "" | index-list connectivity for h/R = 1/48 and beyond (`_compact` output names) |
| `DAMP_BETA`, `TMAX` | 0.5, 4000 | Picard damping (exact, see `jaxfdm_gaslaw/common.py`) and iteration cap |

### Expected numbers

`python sec31/table1.py` prints Table 1 from the shipped files; the published table is
its rounding (gradient ratio = |dL/dtheta|_p / |dL/dtheta|_Kgas on the stable branch at
p = p_f (1 - eps)):

```
                                 uniform, global limit point            zoned, local limit point
h/R                            1/6   1/12   1/24   1/48   1/96  1/192    1/12   1/24   1/48   1/96  1/192
p_f/p*                      1.0225 1.0055 1.0013 1.0003 1.0001 1.0000   0.636  0.646  0.655  0.661  0.663
lambda_min: prescribed x1e-13    3      3      3    0.8    0.4    1.6      57     64     67      9      4
            gas law          0.289  0.285  0.283  0.282  0.282  0.282   0.183  0.156  0.145  0.140  0.138
            vol. constraint  0.289  0.285  0.283  0.282     --     --   0.283  0.246  0.232     --     --
gradient ratio, eps = 1e-3       5      5      6      6     --     --       4      4      4     --     --
                eps = 1e-7     392    431    442    445     --     --     303    319    328     --     --
(1 + S)^2, eps = 1e-3         1202   1158   1147   1144     --     --     158    123    110     --     --
           eps = 1e-7, x1e6     12     11     11     11     --     --     1.3   0.97   0.85     --     --
pole exponent n, prescribed -0.992 -0.994 -0.994 -0.994     --     --  -1.000 -0.998 -0.996     --     --
                 gas law    -0.002 -0.002 -0.002 -0.002     --     --  -0.013 -0.016 -0.025     --     --
S at 6.7 V_f                    --     --     --     --     --     --   -3.10  -2.97  -2.89     --     --
```

Other numbers of Sec. 3.1 and where they come from:

| statement in the paper | script and quantity | value |
|---|---|---|
| discrete limit point converges to the analytic one at O(h^2) | `cap_verify.py FOLD_ONLY=1`, p_f/p* - 1 | 0.0225, 0.0055, 0.0013, 0.0003, ... |
| gas law lifts lambda_min to about 0.28 (uniform) and 0.14 (zoned) on every mesh | `cap_verify.py`, lamG_fold | table above |
| c_phi: the rank-one term is 1.125 (uniform) / about 0.3 (zoned) of the network stiffness | `cap_closure_probe.py`, c_phi | 1.155, 1.132, 1.127, 1.125; 0.394, 0.339, 0.318 |
| identity of Eq. 22 holds to 1e-13 | `cap_verify.py`, column "identity" | <= 1e-13 at every state |
| responses confirmed by FD to 1e-8 (gas law) and 1e-7 / 1e-5 (prescribed) | `cap_verify.py`, columns "dxG FD", "dxP FD" | see log |
| gas-law continuation passes the fold in at most 3 Newton iterations per step | `cap_verify.py`, "[gas law]" block | Newton its max 2 (uniform), 3-4 (zoned path) |
| zoned limit point = bubble on the neck, p_f/(2 N_in/a) | `cap_bubble_check.py` | 1.034, 1.006, 1.000 (1/12, 1/24, 1/48) |
| S approaches -3 past the local fold | `cap_model.py` zoned, `V_END=1.56` | -3.10, -2.97, -2.89 at 6.7 V_f |
| jax_fdm gas-law gradient vs FD through the fold | `jaxfdm_fd_through_fold.py` | worst of the best per station: 2.7e-9 relative at 1/24 (1.2e-11 absolute) |
| adjoint vs dense direct solve | `jaxfdm_adjoint_reference.py` | gas law <= 7e-14; prescribed 1e-14 far, 3e-10 to 1e-9 next to the fold |
| prescribed pressure stalls on the limit point, (1 + S)^2 between 1e11 and 1e19 | `cap_optimise_target.py` at 1/24 | `results/capopt_fold/paper/*.log` |

## Verification of this repository

See `docs/VERIFICATION.md` for the runs made with this exact code on 2026-10-19 and their
agreement with the shipped files and the published table.

## Notes on the numerics

* **One discrete problem, two codes.** `fdm_limit_point` and `jaxfdm_gaslaw` build the same
  A2 triangulation, the same force densities and the same enclosed volume. On a triangulated
  mesh jax_fdm's edge-to-centroid lumped face load equals grad V of the triangulated volume
  (4.5e-17 on this mesh), so both tangents are exactly symmetric and the limit points of the
  two codes agree to 1e-10 in p* (`jaxfdm_fold_ladder.py`).
* **Forward solve.** SciPy side: Newton on the bordered system (volume control), or at fixed p
  (prescribed pressure), or on K~_T through the bordered form of Eqs. 19-20 (gas law). JAX-FDM
  side: damped Picard, beta = 1/2, which is the exact damping of the pressure-volume breathing
  mode of the gas-law fixed-point map (see `jaxfdm_gaslaw/common.py`).
* **Adjoint.** jax_fdm's shipped backward rule (NormalCG at 1e-6) caps gradient accuracy at
  about 3e-7 and squares the condition number; the GMRES rule at 1e-12 of `common.py` is used
  for every number in the paper.
* **Speed-ups.** A hash-map edge-to-face table (always on) and, for h/R >= 1/48, index-list
  connectivity (`JAXFDM_COMPACT=1`) and a sparse K^-1 in the adjoint (`gmres_sparse`); each was
  checked to leave the results unchanged (2e-14 in the equilibria, 1e-8 in gradient norms).
* **Never overwrite.** Scripts write new time-stamped files; `newest(pattern)` in `cap_model.py`
  picks the latest, falling back to the shipped `paper/` files.

## Licence and citation

The code in `fdm_limit_point/`, `jaxfdm_gaslaw/common.py`, `sec31/` and `tests/` is by the
author of the paper. `jaxfdm_gaslaw/patch/` contains modified files of
[jax_fdm](https://github.com/arpastrana/jax_fdm) (MIT licence, Rafael Pastrana). A licence
file for this repository has not been chosen yet.
