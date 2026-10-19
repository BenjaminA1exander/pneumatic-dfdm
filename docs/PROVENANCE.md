# Provenance of the shipped result files

Every file under `results/*/paper/` was produced by the scripts of this repository (under
their original names in the authors' working tree) and is one of the files behind the
numbers and figures of Sec. 3.1 of the paper. They are shipped so that Table 1 and
Fig. 3 can be assembled without re-running the finest meshes, and so that re-runs can be
compared with them. Nothing in `paper/` is ever overwritten by the scripts: new runs
land one level up, with a later time stamp.

## results/sec31/paper

| files | script (env) | used for |
|---|---|---|
| `fold_h{6,12,24,48,96,192}_g1_b1_rb0.5_*.npz` | `cap_verify.py` `FOLD_ONLY=1 CAP=uniform` | Table 1 rows p_f/p* and lambda_min (prescribed, gas law); Fig. 3(e) uniform |
| `fold_h{12,24,48,96,192}_g0.25_b10_rb0.5_w0.125_blend_*.npz` | `cap_verify.py` `FOLD_ONLY=1 CAP=zoned` | same, zoned cap; Fig. 3(e) zoned |
| `verify_h{6,12,24,48}_g1_b1_rb0.5_*.npz` | `cap_verify.py` `CAP=uniform` | Table 1 gradient ratio and (1+S)^2; Fig. 3(d); pole exponents |
| `verify_h{12,24,48}_g0.25_b10_rb0.5_w0.125_blend_*.npz` | `cap_verify.py` `CAP=zoned` | same, zoned cap |
| `closureprobe_h{6,12,24,48}_g1_b1_*.npz`, `closureprobe_h{12,24,48}_g0.25_b10_w0.125_blend_*.npz` | `cap_closure_probe.py` | Table 1 row "vol. constraint"; c_phi of claim (iv) |
| `pin_h24_g1_b1_rb0.5_*.npz`, `pin_h{12,24,48}_g0.25_b10_rb0.5_w0.125_blend_*.npz` | `cap_model.py` (zoned with `V_END=1.56`) | Fig. 3(b,c); S at 6.7 V_f (claim v) |
| `bubblecheck_h{12,24,48}_*.npz` | `cap_bubble_check.py` | the zoned limit point is the bubble on the neck (Sec. 3.1, claim i) |
| `fig2_method_paper.{pdf,png}`, `fig3_verification_paper.{pdf,png}` | `fig2_method.py`, `fig3_verification.py` | Fig. 2 and Fig. 3 as submitted |

Machines: laptop (Intel i7-10750H) for h/R <= 1/48; the 1/96 and 1/192 fold files on the
same laptop (hours) and on the ETH Euler cluster. Dates 2026-10-14 to 2026-10-18 (in the
file names).

## results/jaxfdm/paper

| files | script | used for |
|---|---|---|
| `jaxfdm_fold_h{6,12,24}.npz`, `jaxfdm_fold_h48_compact.npz` | `jaxfdm_fold_ladder.py` | the limit point from JAX-FDM alone; agrees with the SciPy fold to 1e-10 in p* |
| `jaxfdm_fold_spectra_h*.npz` | `jaxfdm_fold_spectra.py` | lambda_min of both adjoint operators at the fold, matrix-free |
| `jaxfdm_pole_h*.npz` | `jaxfdm_pole_exponent.py` | pole exponent n of the gradient norm (claim ii) |
| `limitpoint_smref_tri6_scipyfold.npz`, `limitpoint_smref_tri12.npz`, `limitpoint_smref_tri24_delftblue.npz` | `jaxfdm_adjoint_reference.py` (uniform) | adjoint accuracy against the dense Sherman-Morrison reference: 1e-12 away from the fold, 1e-9 next to it |
| `limitpoint_smref_zoned_g0.25_b10_w0.125_blend_tri{12,24}.npz` | `jaxfdm_adjoint_reference.py` `ZONED=1` | same, zoned cap |
| `fd_through_fold_{uniform,zoned}_tri{12,24}_*.npz` + logs | `jaxfdm_fd_through_fold.py` (Euler) | FD of the nonlinear solve against the gas-law gradient through the fold: 1e-11 uniform, 7e-10 zoned (absolute) |

The `_compact` files were produced with `JAXFDM_COMPACT=1` (index-list connectivity); the
`_delftblue` file on the TU Delft DelftBlue cluster. Two `fd_through_fold_*_tri24` files
exist per cap because the job was run twice; both are complete and agree.

## results/capopt_fold/paper

`capopt_fold_{uniform,zoned_blend}_h24_s{0.4,0.7,0.9895}_eps0.05.npz` and the job logs:
`cap_optimise_target.py` at h/R = 1/24 on Euler (2026-10-18), three starts per cap. Fig. 3(f)
shows the four runs that stop on the limit point (uniform 0.7 and 0.9895, zoned 0.4 and 0.7).
The uniform 0.4 run creeps without reaching the fold within 1000 iterations and the zoned
0.9895 run jumps the fold and reaches the target on the unstable branch; both are kept here.
