r"""config.py -- every parameter of the FDM limit-point study in one place.

Units are fixed by R = 1 and q = 1, so N0 = sqrt(3) q = sqrt(3) and the analytic
limit point sits at p* = 2 N0 / R = 2 sqrt(3), V* = (2/3) pi R^3.

Nothing in this study is tuned to the expected answer. The only quantity that is
allowed to move in response to a measurement is `trim` (the lattice removal
threshold), and only if the generated mesh fails the 20 degree minimum-angle
requirement -- which is a mesh-quality gate, not a result.
"""
from dataclasses import dataclass, asdict
import math


@dataclass
class Config:
    # --- model -------------------------------------------------------------
    R: float = 1.0                  # rim radius
    q: float = 1.0                  # uniform force density (q_mode="uniform")

    # --- meshes ------------------------------------------------------------
    levels: tuple = (6, 12, 24, 48)     # refinement: h/R = 1/level (the table)
    fig_levels: tuple = (4, 8, 16)      # the three branches drawn in panel (b)
    q_modes: tuple = ("uniform", "cotan")
    trim: float = 0.5                   # drop lattice points with r > R - trim*h
    min_angle_target: float = 20.0      # mesh-quality gate, degrees
    trim_ladder: tuple = (0.5, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65)

    # --- volume-controlled continuation ------------------------------------
    v_start_frac: float = 0.02      # first target, in units of V*
    v_end_frac: float = 1.5         # last target, in units of V*
    dv_frac: float = 0.02           # initial AND maximum step, in units of V*
    dv_min_frac: float = 1.0e-6     # give up below this
    newton_max: int = 20
    newton_grow: int = 3            # <= this many iterations -> dV *= 1.5
    newton_shrink: int = 8          # >  this many iterations -> dV *= 0.5
    tol_r: float = 1.0e-10          # ||r||_2 < tol_r * q * R
    tol_v: float = 1.0e-12          # |V - Vbar| < tol_v * V*

    # --- spectra ------------------------------------------------------------
    n_eig: int = 4                  # lowest eigenvalues of K_T phi = mu M phi
    eig_shift: float = -0.1         # shift-invert target, in units of q / R^2
    eig_stride: int = 1             # compute spectra every n-th converged point

    # --- fold location ------------------------------------------------------
    fold_tol: float = 1.0e-8        # bisect on Vbar until |dp/dVbar| < fold_tol
    fold_max_bisect: int = 80

    # --- checks / output ----------------------------------------------------
    fd_seed: int = 0
    outdir: str = "results/fdm_limit_point"
    fig_qmode: str = "uniform"      # q_mode drawn in the paper figure

    # --- derived ------------------------------------------------------------
    @property
    def N0(self) -> float:
        """Continuum membrane tension of a uniform-q equilateral lattice."""
        return math.sqrt(3.0) * self.q

    @property
    def p_star(self) -> float:
        return 2.0 * self.N0 / self.R

    @property
    def V_star(self) -> float:
        return (2.0 / 3.0) * math.pi * self.R ** 3

    def as_dict(self):
        return asdict(self)
