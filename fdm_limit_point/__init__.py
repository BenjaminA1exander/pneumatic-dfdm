r"""fdm_limit_point -- reference implementation of the pneumatic cap of Sec. 2.1.2.

A force-density net on an equilateral (A2) triangulation of a disk, pinned on its
rim and loaded by a follower pressure that is the exact gradient of the enclosed
volume (Eq. 4 of the paper).  The package provides

    config.py        every parameter of the study (R = 1, q = 1, so p* = 2 sqrt(3))
    mesh.py          the A2 lattice clipped to the disk, rim nodes on the circle
    analytic.py      the constant-tension spherical cap (Eqs. 1-3)
    model.py         K, D, V, grad V, grad^2 V, residual r and tangent K_T (Eqs. 4-6)
    continuation.py  volume-controlled Newton (the bordered system of Eq. 10),
                     branch tangent, spectra of K_T, and location of the limit point
    zones.py         the zoned force densities of the "local fold" cap (Sec. 3.1)

The scripts in ../sec31 build the three closures of Sec. 2.2.2 on top of these
operators:  prescribed pressure (K_T), gas law (K_T + (p/V) g g^T, Eq. 18) and the
volume constraint (the bordered matrix).  Everything here is NumPy/SciPy; JAX is
used only for the exact derivatives of the volume functional.
"""
from .config import Config                                        # noqa: F401
