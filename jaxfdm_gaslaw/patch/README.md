# Gas-law patch for jax_fdm 0.10.0

JAX-FDM (Pastrana et al., MIT licence) solves force-density equilibrium as a fixed
point and differentiates it implicitly. The paper changes one thing: the load
function. Under the gas law the face pressure is `p = K_gas / V(x)` (Eq. 16), so the
load, and with it the adjoint operator, gains the rank-one term of Eq. 18.

The three files in this folder are drop-in replacements for the files of the PyPI
release 0.10.0:

| file | change |
|---|---|
| `equilibrium/models.py` | `global_load_fn` hook on `EquilibriumModel` and the class `ConstantMassModelSparse` (the gas-law load, differentiated by the implicit adjoint) |
| `equilibrium/loads.py` | `face_xyz` fix for padded triangular faces of mixed tri/quad meshes (no effect on the all-triangle meshes of Sec. 3.1) |
| `equilibrium/solvers/fixed_point.py` | host-side tally of forward and adjoint solves that stop without converging; printed at exit |

`jax_fdm-0.10.0-gaslaw.diff` is the unified diff. `sha256_pypi_0.10.0.txt` and
`sha256_patched.txt` identify the pristine and the patched versions.

```
pip install jax_fdm==0.10.0 --no-deps      # the environment file pins the dependencies
python jaxfdm_gaslaw/patch/apply_patch.py  # copies the three files, keeps <file>.orig
python jaxfdm_gaslaw/patch/apply_patch.py --check
```
