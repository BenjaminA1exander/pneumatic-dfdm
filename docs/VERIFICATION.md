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
| jaxfdm_fd_through_fold | 1/6, 1/12 | best FD agreement per state, worst over the 10 states through the fold: 2.6e-9 (1/6), 1.3e-9 (1/12) relative | Euler 1/12 file: 1.3e-9, the same digits at every state |
| jaxfdm_pole_exponent | 1/12 | n_presc -0.995, n_gas -0.0013; gas-law norms vs dense reference 1.3e-8 | same |
| jaxfdm_adjoint_reference | 1/12 | gas-law adjoint vs dense: worst 4.6e-14; prescribed vs Sherman-Morrison: 2.7e-14 far from the fold, 2.7e-10 next to it | Table 1 draft: 0.5e-13 and 4e-10 |
| jaxfdm_adjoint_reference | 1/12 zoned | gas-law adjoint vs dense: median 4.3e-13, worst 4.7e-13; prescribed vs Sherman-Morrison: 6.9e-14 far from the fold, 8.0e-10 next to it; rank-one identity 6.7e-15 | shipped file: gas-law arrays agree to 6e-15 absolute, prescribed-pressure arrays within 0.3 decades away from V_f |
| jaxfdm_fd_through_fold | 1/12 zoned | worst of the best FD agreement 4.8e-8 (1.3e-9 .. 4.8e-8 over the 10 states) | Euler file: 4.8e-8, the same digits at every state |

A note on the finite-difference check. The first laptop run of `jaxfdm_fd_through_fold.py`
gave 1e-8 .. 9e-7, a constant offset at every step size, because the refactored script no
longer switched the adjoint linear solver to the GMRES rule at rtol 1e-12
(`jaxfdm_gaslaw/common.py`); jax_fdm's shipped rule (NormalCG at 1e-6) caps the gradient
at about 1e-6 relative. The script now calls `use_adjoint("gmres")` itself, and the numbers
above are from that version. The prescribed-pressure columns of the adjoint reference are
condition-number limited (error ~ kappa x tolerance) and the station on V_f itself is
round-off; `compare_to_paper.py` leaves that station out and compares the rest in decades.

## Claim (vi), Fig. 3(f): design under each closure  (`run_capopt.sh`, LEVEL=6, uniform, 30 min)

Three starts (s = 0.4, 0.7, 0.9895 of V_f) towards the same target just past its own limit
point (V_t / V_f,t = 1.05), 400 design variables, 1000 L-BFGS iterations at most:

| start | prescribed pressure | gas law |
|---|---|---|
| 0.4 | stalls after 135 its at L = 7.5e-3 (line search fails; 349 forward failures); end (1+S)^2 = 6e15 | L = 3.1e-11 after 1000 its, nodes within 2.2e-5 R of the target |
| 0.7 | stalls after 135 its at L = 6.8e-3; (1+S)^2 = 3e11 | L = 1.3e-10, nodes within 3.6e-5 R |
| 0.9895 | stalls after 118 its at L = 7.0e-3; (1+S)^2 = 5e11 | L = 4.8e-10, nodes within 8.5e-5 R |

This is the behaviour the paper reports at 1/24 (`results/capopt_fold/paper/*.log`:
prescribed pressure ends at L = 2.9e-2 .. 2.5e-3 with (1+S)^2 up to 7e11, the gas law at
3e-10 .. 2e-9). The 1/24 runs take hours each and are not repeated here.

## Not re-run here

h/R = 1/48 and finer (hours to days; the shipped files were produced on the laptop and on
Euler, see `docs/PROVENANCE.md`), the 1/24 dense reference and FD check (16 GB), and the
1/24 design runs of Fig. 3(f) (hours each on Euler).

## Differences `compare_to_paper.py` reports that are not discrepancies

- `pin_h24_*`: the adaptive continuation takes a different step sequence (150 and 78 states
  against 180 and 301), so the `rows` arrays differ in shape and the final state differs by
  the last step (1 to 8 %); the curves coincide. `belt_w` in the uniform-cap file is the
  stored default of the zoned-cap belt width (0.125 now, 0.0625 when the shipped file was
  made); the uniform cap does not use it.
- `fd_through_fold_*`: the individual finite-difference errors scatter by up to 10 % between
  machines (round-off of the forward solves); the best agreement per state is what the
  paper quotes, and that matches.
- `limitpoint_smref_*`: the prescribed-pressure columns, see the note above.
