r"""jaxfdm_gaslaw -- the gas-law closure of Sec. 2.2 inside JAX-FDM.

    patch/      the three jax_fdm 0.10.0 files that add the gas-law load
                (ConstantMassModelSparse) and the installer apply_patch.py
    common.py   the pneumatic cap as a JAX-FDM problem: mesh, damped Picard
                forward solve, adjoint rules, and the two tangent operators

Import `jaxfdm_gaslaw.common` only from a script that has put the repository
root on sys.path (every script in ../sec31 does), because it also needs the
reference mesh of ../fdm_limit_point.
"""
