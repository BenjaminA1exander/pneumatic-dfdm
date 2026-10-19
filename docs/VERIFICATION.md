# Verification of this repository (2026-10-19)

Everything below was run from this repository, with the code as committed, on the authors'
laptop (Intel i7-10750H, Windows 11, the `jaxfdm` conda environment of `environment.yml`).
The logs are in `docs/verification_logs/`. `sec31/compare_to_paper.py` compares every
re-run result file with the shipped file of the same kind, cap and mesh.

## Tests

`python tests/test_fdm_limit_point.py`: 9 of 9 pass (mesh quality, analytic cap, grad V and
grad^2 V against finite differences, K_T symmetry to 1e-14, branch tangent against converged
states, sparse against dense spectra, the published limit point at h/R = 1/6, Dirichlet and
small-volume convergence orders, zone fractions).

## SciPy side: Table 1, Fig. 2, Fig. 3  (`run_sec31.sh`, LEVELS="6 12 24", 30 min)

| re-run file | shipped file | largest relative difference |
|---|---|---|
| fold_h{6,12,24}, uniform | same | 0 (bit-identical) |
| fold_h{12,24}, zoned | same | 4.5e-14 |
| verify_h{6,12,24}, uniform | same | 0 |
| verify_h{12,24}, zoned | same | 4.8e-9 (1/12), 2.1e-11 (1/24) |
| closureprobe, all five | same | 0 |
| bubblecheck_h{12,24} | same | 0 |
| pin_h24 (both caps) | same curve, different step sequence (adaptive continuation; 150 and 78 states against 180 and 301) | not comparable point by point |

`table1.py` therefore prints the same column values at 1/6, 1/12 and 1/24 as from the shipped
data (`docs/verification_logs/table1.log`), including the pole exponents and S at 6.7 V_f on
the zoned cap (-2.97). The per-mesh summary blocks of `cap_verify.py` also reproduce the
text numbers: the gas-law continuation passes the fold with at most 2 (uniform) or 3 (zoned)
Newton iterations per step; the identity of Eq. 22 holds to 1e-15..1e-13; the adjoint
gradients agree with finite differences of the nonlinear solve to 3e-9..1e-5 relative
(the finite-difference reference, not the adjoint, sets the floor at 1/24).

`fig2_method.py` and `fig3_verification.py` regenerate Fig. 2 and Fig. 3
(`results/sec31/fig2_method.png`, `fig3_verification_h24_w0.125_blend.png`); the panels are
visually identical to `results/sec31/paper/fig{2,3}_*_paper.png`.

## JAX-FDM side  (`run_jaxfdm.sh`, LEVELS="6 12")

| script | mesh | result | shipped / paper |
|---|---|---|---|
| jaxfdm_fold_ladder | 1/6, 1/12 | p*_h 3.5420172417, 3.4831823070; SciPy fold 4.4e-11 away | same to all printed digits |
| jaxfdm_fold_spectra | 1/6, 1/12 | gas law 0.289364, 0.284993; prescribed 1.9e-9, 6.5e-9 | same |
| jaxfdm_pole_exponent | 1/6 | n_presc -0.994, n_gas -0.0012; norms vs dense reference 8.7e-9 (gas), 1.3e-4 (prescribed, station placement) | same |
| jaxfdm_adjoint_reference | 1/6 | gas-law adjoint vs dense: worst 3.7e-14; prescribed vs Sherman-Morrison: 9e-15 far from the fold, 3.2e-10 next to it (V_f itself excluded) | Table 1 draft: 0.4e-13 and 3e-10 |
| jaxfdm_fd_through_fold | 1/6 | best FD agreement per state 1e-8 .. 5e-7 relative over 10 states through the fold | the paper's 1/12 and 1/24 values are the Euler runs in results/jaxfdm/paper |

(The 1/12 dense reference, FD check, the zoned 1/12 runs and the claim (vi) design runs at
1/6 are appended below when finished.)

## Not re-run here

h/R = 1/48 and finer (hours to days; the shipped files were produced on the laptop and on
Euler, see `docs/PROVENANCE.md`), the 1/24 dense reference and FD check (16 GB), and the
1/24 design runs of Fig. 3(f) (hours each on Euler).
