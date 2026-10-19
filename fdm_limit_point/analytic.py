r"""analytic.py -- the continuous constant-tension spherical cap on a rim of radius R.

The continuum limit of the uniform-q lattice is a membrane with strain energy
W = (N0/2)(lam1^2 + lam2^2) per unit REFERENCE area, i.e. the Dirichlet energy of
the map from the flat disk to the deformed surface. Its stationary points are
harmonic maps; the conformal (inverse-stereographic) map from the disk onto a
spherical cap is harmonic, and for a conformal map lam1 = lam2 = lam the Cauchy
membrane stress is

    N = (1/lam^2) * N0 * F F^T = N0 * P_tangent,

isotropic of magnitude N0 whatever the stretch. So the constant-tension spherical
cap is an EXACT solution of the continuum problem for every polar half-angle
theta, and Laplace on a sphere of radius rho = R / sin(theta) gives

    p(theta) = 2 N0 / rho = (2 N0 / R) sin(theta),

whose maximum over the branch is the hemisphere theta* = pi/2:

    p* = 2 N0 / R,  V* = (2/3) pi R^3,  H* = R.

Cap geometry with base radius R and apex height H = R tan(theta/2):

    V(theta) = (pi/6) H (3 R^2 + H^2)
             = (pi/6) R^3 tan(theta/2) (3 + tan^2(theta/2)).
"""
import numpy as np


def pressure(theta, N0, R):
    return (2.0 * N0 / R) * np.sin(theta)


def volume(theta, R):
    t = np.tan(0.5 * theta)
    return (np.pi / 6.0) * R ** 3 * t * (3.0 + t ** 2)


def apex_height(theta, R):
    return R * np.tan(0.5 * theta)


def theta_of_volume(V, R):
    """Invert V(theta) for theta in (0, pi). Cubic in t = tan(theta/2), one real
    positive root, taken by Cardano rather than by iteration."""
    V = np.asarray(V, dtype=float)
    c = 6.0 * V / (np.pi * R ** 3)              # t^3 + 3t - c = 0
    d = np.sqrt((0.5 * c) ** 2 + 1.0)
    t = np.cbrt(0.5 * c + d) + np.cbrt(0.5 * c - d)
    return 2.0 * np.arctan(t)


def branch(V, N0, R):
    """Analytic p(V) along the constant-tension cap branch."""
    return pressure(theta_of_volume(V, R), N0, R)
