r"""Independent checks of the reference implementation (fdm_limit_point) against
finite differences, dense linear algebra, the analytic cap and the published numbers.

    python tests/test_fdm_limit_point.py      (no pytest needed; ~30 s)
    python -m pytest tests                    (if pytest is installed)

Nothing here influences a result: every check re-derives a quantity by a different
route (finite differences, a dense eigensolver, the analytic bubble) so that a
disagreement is visible rather than absorbed. The cross-mesh checks pass on an
OBSERVED CONVERGENCE ORDER, not on a tolerance that could be loosened.
"""
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fdm_limit_point.config import Config                                   # noqa: E402
from fdm_limit_point import mesh as meshmod, analytic                       # noqa: E402
from fdm_limit_point.model import Model                                     # noqa: E402
from fdm_limit_point.continuation import sweep, locate_fold, newton, spectra, bordered  # noqa: E402
from fdm_limit_point.zones import edge_fractions, zone_q                    # noqa: E402

J01_SQ = 5.783185962946785          # first zero of J0, squared (Dirichlet eigenvalue of the unit disk)
_CACHE = {}


def case(level):
    """Mesh, model, branch and fold of the uniform cap at h/R = 1/level (cached)."""
    if level not in _CACHE:
        cfg = Config()
        msh = meshmod.build(level, cfg)
        model = Model(msh, cfg, "uniform")
        model.reference()
        rows, states = sweep(model, cfg, log=lambda *a: None)
        fold = locate_fold(model, cfg, states, log=lambda *a: None)
        _CACHE[level] = (cfg, msh, model, rows, states, fold)
    return _CACHE[level]


def central(f, x, e, step):
    return (f(x + step * e) - f(x - step * e)) / (2.0 * step)


def richardson(f, x, e, step):
    """V is EXACTLY cubic in the coordinates, so one Richardson pair of central
    differences cancels the truncation term identically; a LARGE step then minimises
    round-off (the only error left)."""
    return (4.0 * central(f, x, e, 0.5 * step) - central(f, x, e, step)) / 3.0


# ----------------------------------------------------------------------------- mesh and analytic cap
def test_mesh_is_an_a2_lattice_of_good_quality():
    for level in (6, 12):
        cfg, msh, *_ = case(level)
        assert msh["min_angle"] >= cfg.min_angle_target
        assert msh["n_rim"] == round(2 * math.pi * level)
        assert msh["trim"] == cfg.trim                      # no lattice trimming was needed
        assert abs(msh["h"] - cfg.R / level) < 1e-15


def test_analytic_cap():
    cfg = Config()
    th = np.linspace(0.05, 3.0, 50)
    V = analytic.volume(th, cfg.R)
    assert np.allclose(analytic.theta_of_volume(V, cfg.R), th, atol=1e-12)
    p = analytic.pressure(th, cfg.N0, cfg.R)
    assert p.max() <= cfg.p_star and abs(th[np.argmax(p)] - np.pi / 2) < 0.07   # maximum at the hemisphere
    assert abs(analytic.pressure(np.pi / 2, cfg.N0, cfg.R) - cfg.p_star) < 1e-14   # p* = 2 N0 / R (Eq. 3)
    assert abs(analytic.volume(np.pi / 2, cfg.R) - cfg.V_star) < 1e-14


# ----------------------------------------------------------------------------- operators
def test_grad_V_against_richardson_fd():
    cfg, msh, model, *_ = case(6)
    rng = np.random.default_rng(0)
    xu = model.x_flat + 0.1 * cfg.R * rng.standard_normal(model.ndof)
    g = model.grad_V(xu)
    err = 0.0
    for i in range(model.ndof):                              # componentwise, 327 dof
        e = np.eye(1, model.ndof, i).ravel()
        err = max(err, abs(richardson(model.volume, xu, e, 0.1 * cfg.R) - g[i]) / abs(g[i]))
    assert err < 1e-7, err


def test_hess_V_against_fd_and_K_T_symmetry():
    cfg, msh, model, *_ = case(6)
    rng = np.random.default_rng(0)
    xu = model.x_flat + 0.1 * cfg.R * rng.standard_normal(model.ndof)
    H = model.hess_V(xu)
    err = 0.0
    for _ in range(8):                                       # grad V is exactly quadratic: no truncation error
        e = rng.standard_normal(model.ndof)
        e /= np.linalg.norm(e)
        fd = central(model.grad_V, xu, e, 0.1 * cfg.R)
        err = max(err, np.linalg.norm(fd - H @ e) / np.linalg.norm(H @ e))
    assert err < 1e-7, err
    KT = model.tangent(xu, 1.0, H)
    assert abs(KT - KT.T).max() / abs(KT).max() < 1e-14


def test_branch_tangent_against_fd_of_converged_states():
    cfg, msh, model, rows, states, fold = case(6)
    st = states[len(states) // 2]
    d = 1e-4 * cfg.V_star
    xs = []
    for s in (+1, -1):
        ok, xn, pn, _it, _ = newton(model, st["xu"] + s * d * st["xdot"], st["p"] + s * d * st["pdot"],
                                    st["Vbar"] + s * d, cfg)
        assert ok
        xs.append((xn, pn))
    fd_x = (xs[0][0] - xs[1][0]) / (2.0 * d)
    fd_p = (xs[0][1] - xs[1][1]) / (2.0 * d)
    assert np.linalg.norm(fd_x - st["xdot"]) / np.linalg.norm(st["xdot"]) < 1e-6
    assert abs(fd_p - st["pdot"]) / (cfg.p_star / cfg.V_star) < 1e-6


def test_sparse_spectra_against_dense():
    cfg, msh, model, rows, states, fold = case(6)
    st = states[len(states) // 2]
    ok, xu, p, _it, state = newton(model, st["xu"], st["p"], st["Vbar"], cfg)
    state["xu"] = xu
    mu, *_ = spectra(model, state, cfg)
    w = eigh(state["KT"].toarray(), model.M3.toarray(), eigvals_only=True)[:cfg.n_eig]
    assert np.abs(np.sort(mu) - w).max() / abs(w).max() < 1e-8


# ----------------------------------------------------------------------------- the limit point (Table 1, h/R = 1/6)
def test_limit_point_matches_table_1():
    cfg, msh, model, rows, states, fold = case(6)
    assert fold is not None
    assert abs(fold["p"] / cfg.p_star - 1.02249) < 2e-5           # Table 1: p_f/p* = 1.0225
    assert abs(fold["V"] / cfg.V_star - 1.02230) < 2e-5
    assert abs(fold["z_apex"] / cfg.R - 1.03023) < 2e-5
    st = fold["state"]
    KT, g, K3 = st["KT"].toarray(), st["gV"], model.K3.toarray()
    lam_P = eigh(KT, K3, eigvals_only=True)                      # K_T phi = lambda K phi (Eq. 14)
    # claim (i): K_T singular at the pressure maximum. The fold is located to |dp/dVbar| < 1e-8
    # (cfg.fold_tol), which leaves lambda_min of this order; cap_verify.py bisects to 1e-13.
    assert abs(lam_P[0]) < 1e-7
    c = fold["p"] / fold["V"]
    lam_G = eigh(KT + c * np.outer(g, g), K3, eigvals_only=True)  # gas law, Eq. 18
    assert abs(lam_G[0] - 0.289364) < 2e-6                       # Table 1: gas-law lambda_min = 0.289
    # the critical mode changes the volume (Eq. 10: grad V^T phi = 1 after scaling)
    phi = eigh(KT, K3)[1][:, 0]
    assert abs(g @ phi) / (np.linalg.norm(g) * np.linalg.norm(phi)) > 0.5
    # the bordered Jacobian stays regular while K_T is singular
    Jb = bordered(st["KT"], g).toarray()
    s = np.linalg.svd(Jb, compute_uv=False)
    sK = np.linalg.svd(KT, compute_uv=False)
    assert s.min() / s.max() > 1e3 * sK.min() / sK.max()


# ----------------------------------------------------------------------------- cross-mesh: observed orders
def orders(h, e):
    h, e = np.asarray(h, float), np.asarray(e, float)
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def test_dirichlet_limit_and_small_volume_converge():
    levels = (6, 12)
    hs, mu1, dev = [], [], []
    for L in levels:
        cfg, msh, model, rows, *_ = case(L)
        hs.append(msh["h"])
        mu1.append(rows[0]["mu1"] * cfg.q / cfg.N0)             # mu1 R^2 / N0 at p = 0
        v = np.array([r["V"] for r in rows])
        p = np.array([r["p"] for r in rows])
        m = (v >= 0.02 * cfg.V_star) & (v <= 0.20 * cfg.V_star)
        pa = analytic.branch(v[m], cfg.N0, cfg.R)
        dev.append(np.max(np.abs(p[m] - pa) / pa))
    e = np.abs(np.array(mu1) - J01_SQ) / J01_SQ               # N0 = sqrt(3) q: tangent -> N0 x Dirichlet Laplacian
    assert e[1] < e[0] and orders(hs, e)[-1] >= 1.5
    assert dev[1] < dev[0] and orders(hs, dev)[-1] >= 1.5     # discrete p(V) -> analytic cap at O(h^2)


# ----------------------------------------------------------------------------- zones
def test_zone_fractions_and_blend():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (200, 2))
    b = a + rng.uniform(-0.2, 0.2, (200, 2))
    fr = edge_fractions(a, b, 0.5, 0.125)
    assert np.allclose(fr.sum(1), 1.0)
    # an edge inside one zone gets that zone's q under either scheme
    q_in, _ = zone_q(np.array([[0.1, 0.0]]), np.array([[0.15, 0.05]]), 0.5, 0.125, 0.25, 10.0)
    assert abs(q_in[0] - 0.25) < 1e-15
    q_belt, _ = zone_q(np.array([[0.5, 0.0]]), np.array([[0.5, 0.05]]), 0.5, 0.125, 0.25, 10.0)
    assert abs(q_belt[0] - 10.0) < 1e-15
    # a chord between two belt points dips into the soft disk: the blend sees that, the midpoint rule would not
    q_chord, _ = zone_q(np.array([[0.5, 0.0]]), np.array([[0.0, 0.5]]), 0.5, 0.125, 0.25, 10.0)
    assert 0.25 < q_chord[0] < 10.0
    q_uni, _ = zone_q(a, b, 0.5, 0.125, 1.0, 1.0)
    assert np.allclose(q_uni, 1.0)


if __name__ == "__main__":
    import time
    names = [n for n in dir() if n.startswith("test_")]
    failed = 0
    for n in names:
        t = time.time()
        try:
            globals()[n]()
            print(f"PASS  {n}  ({time.time() - t:.1f} s)")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
