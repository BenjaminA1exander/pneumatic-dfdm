r"""common.py -- the pneumatic cap of Sec. 3.1 as a JAX-FDM problem, under both closures.

WHAT THIS FORMALISES (Sec. 2.2 of the paper)
--------------------------------------------
jax_fdm solves static equilibrium as the fixed point of one force-density update

    x = Y(x, theta) = K(theta)^-1 (P(x) - D x_s)                            (Eq. 11)

over the free-node coordinates x. For a follower pressure the load P depends on x
through the face normals. The implicit adjoint behind `jax.grad` is a linear solve
against the tangent operator

    A = I - dY/dx = K^-1 (K - dP/dx) = K^-1 K_T                            (Eq. 14)

Two closures differ only in what dP/dx contains.

  PRESCRIBED PRESSURE.  p is a parameter, held fixed:  dP/dx = p H (follower load).
  At the pressure limit point K_T is singular (Eq. 10), so sigma_min(A) -> 0: the
  forward Picard stalls and the adjoint has no unique solution (Sec. 2.2).

  GAS LAW (this work).  p = K_gas / V(x) with the gas content K_gas = p V fixed
  (Eq. 15). Differentiating the closure adds the rank-one term of Eq. 17-18:

      dP/dx = p H + grad V dp/dx,   dp/dx = -(p/V) grad V^T,

  so the tangent becomes K~_T = K_T + (p/V) grad V grad V^T, non-singular at a
  simple limit point (Sec. 2.2.1). Only the load function changes; the fixed-point
  solver and the implicit adjoint are those of jax_fdm.

Both operators can be evaluated at the SAME converged state, so any difference is a
property of the closure, not of the solver.

HOW THE GAS LAW ENTERS jax_fdm
------------------------------
`ConstantMassModelSparse` (patch/models.py) recomputes the face load as
p = sign * gas_k / V(xyz) inside the load function that both the forward iteration
and the custom_vjp adjoint differentiate. `faces_volume` is the triangle list for the
enclosed volume (signed projected-area form, the same V as fdm_limit_point.model on
a triangulated surface).

WHY THE MESH IS A TRIANGULATION
-------------------------------
jax_fdm lumps a follower face pressure edge-to-centroid. On a QUAD that lumped load is
not the gradient of the triangulated volume, so K_T is not symmetric (a 4 % mismatch
on the drop-stitch tank of Sec. 3.3.1). On a TRIANGLE each node receives exactly A/3
of the face area vector, which IS dV/dx_i of the triangulated volume: measured on the
cap, the lumped load and grad V agree to 4.5e-17. So on the all-triangle cap mesh the
jax_fdm load is exactly conservative, both tangents are exactly symmetric, and the
gas-law term is an exactly symmetric PSD rank-one update. The mesh is the A2 lattice
of fdm_limit_point.mesh, so every jax_fdm result here can be compared with the
SciPy volume-control study on the SAME discrete problem.

THE ADJOINT RULE
----------------
jax_fdm's shipped backward rule solves the adjoint system with NormalCG at rtol 1e-6.
The normal equations SQUARE the condition number of A -- the one quantity this paper
is about -- and 1e-6 caps adjoint-vs-finite-difference agreement at ~3e-7 whatever
the forward tolerance. `make_bwd` rebuilds the same rule with a choice of lineax
solver and tolerance; GMRES / BiCGStab / NormalCG at rtol 1e-12 agree to 12 digits.
`use_adjoint("gmres")` is used by the dense-reference check, `gmres_sparse` (K^-1 by
jax_fdm's own sparse solve) by the pole-exponent script at the finer meshes.

THE FORWARD SOLVER: DAMPED PICARD, beta = 1/2
---------------------------------------------
For the gas-law model the Jacobian of the Picard map carries exactly ONE eigenvalue at
-1 whenever the supports are coplanar: the pressure-volume breathing mode (p = K/V
answers a volume perturbation with the opposite sign and equal magnitude), so the
undamped iteration oscillates with period two and never converges. The damped map
x <- (1-b) x + b Y(x) has Jacobian (1-b) I + b J, which is ZERO on that mode at
b = 1/2. DAMP_BETA = 0.5 is therefore exact, not tuned. The fixed point and its
derivative are unchanged, and the adjoint operator I - J has eigenvalue 2 there, so
the adjoint is unaffected.

ENVIRONMENT SWITCHES
--------------------
    TRI_LEVEL        h/R = 1/TRI_LEVEL (default 6)
    Q_EDGE_TRI       uniform seed force density (default 1.0, so p* R/N0 = 2)
    DAMP_BETA        Picard damping (default 0.5, see above)
    TMAX             iteration cap of the fixed-point solve (compile-time bound)
    JAXFDM_COMPACT   1 = index-list connectivity instead of the three dense tables
                     (needed at h/R = 1/48 and beyond; identical results to 2e-14)
    XLA_FLAGS        set here to disable XLA constant folding of the index tables,
                     which otherwise takes minutes at h/R = 1/24 and beyond
"""
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_FLAGS", "--xla_disable_hlo_passes=constant_folding")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax import vjp
from jax.experimental.sparse import JAXSparse
from lineax import FunctionLinearOperator, linear_solve, NormalCG, GMRES, BiCGStab
from equinox.internal import while_loop

from jax_fdm.datastructures import FDMesh
try:
    from jax_fdm.equilibrium import (
        EquilibriumModelSparse,
        ConstantMassModelSparse,
        EquilibriumMeshStructureSparse,
    )
except ImportError as exc:                                     # pragma: no cover
    raise ImportError(
        "ConstantMassModelSparse is missing: the installed jax_fdm is not patched. "
        "Run  python jaxfdm_gaslaw/patch/apply_patch.py  (see jaxfdm_gaslaw/patch/README.md)."
    ) from exc
from jax_fdm.equilibrium.states import EquilibriumParametersState
from jax_fdm.equilibrium.solvers import solver_forward
from jax_fdm.equilibrium.solvers import fixed_point as fp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fdm_limit_point import mesh as _refmesh                   # noqa: E402
from fdm_limit_point.config import Config                      # noqa: E402


# =============================================================================
# The cap: A2 triangulation of the unit disk, rim fixed, shallow seed bulge
# =============================================================================

R_BASE = 1.0
SEED_APEX_Z = 0.30          # shallow seed so the seed enclosed volume V0 > 0
Z_TARGET = 0.8              # shape-loss target height, L = mean (z_i - 0.8 R)^2
TRI_LEVEL = int(os.environ.get("TRI_LEVEL", 6))          # h/R = 1/TRI_LEVEL
TAG = f"_tri{TRI_LEVEL}"


def out_path(stem, where):
    """Output name for the active mesh."""
    return os.path.join(where, f"{stem}{TAG}.npz")


def scipy_fold(level=None, zoned=False, zone_tag="_w0.125_blend"):
    """The limit point of the SAME discrete cap located by the independent SciPy code
    (sec31/cap_verify.py with FOLD_ONLY=1): the newest fold_h<L>_... file in
    results/sec31 or results/sec31/paper, or None. Keys: Vf, pf, z_apex, lamP_fold,
    lamG_fold, n_dof."""
    import glob
    level = TRI_LEVEL if level is None else level
    d = os.path.join(ROOT, "results", "sec31")
    pat = (f"fold_h{level}_g0.25_b10_rb0.5{zone_tag}_2*.npz" if zoned
           else f"fold_h{level}_g1_b1_rb0.5_2*.npz")
    f = sorted(glob.glob(os.path.join(d, pat)) + glob.glob(os.path.join(d, "paper", pat)),
               key=os.path.basename)
    return np.load(f[-1]) if f else None


def build_tri():
    """A2 equilateral triangulation of the disk (fdm_limit_point.mesh), seeded with a
    shallow paraboloid bulge. The faces handed to FDMesh are the triangles themselves,
    so the lumped follower load equals grad V exactly (see the module docstring)."""
    cfg = Config(R=R_BASE)
    msh = _refmesh.build(TRI_LEVEL, cfg)
    verts = msh["xyz"].copy()
    r = np.hypot(verts[:, 0], verts[:, 1])
    verts[:, 2] = SEED_APEX_Z * (1.0 - (r / R_BASE) ** 2)
    verts[msh["fixed"], 2] = 0.0
    tris = np.asarray(msh["tris"], dtype=int)
    faces = [[int(a), int(b), int(c)] for a, b, c in tris]
    return (verts, faces, tris, [int(k) for k in msh["fixed"]], int(msh["apex"]))


def enclosed_volume_np(coords, faces_tri):
    """Signed projected-area enclosed volume; matches ConstantMassModelSparse."""
    vi, vj, vk = coords[faces_tri[:, 0]], coords[faces_tri[:, 1]], coords[faces_tri[:, 2]]
    axy = 0.5 * ((vj[:, 0] - vi[:, 0]) * (vk[:, 1] - vi[:, 1])
                 - (vk[:, 0] - vi[:, 0]) * (vj[:, 1] - vi[:, 1]))
    zc = (vi[:, 2] + vj[:, 2] + vk[:, 2]) / 3.0
    return abs(float(np.sum(axy * zc)))


# =============================================================================
# Two jax_fdm speed-ups (validated to leave every result unchanged)
# =============================================================================

# A1. jax_fdm's Mesh._edges_faces compares every edge with every face in Python,
# O(E F): ~1e9 iterations at h/R = 1/48, hours before the first solve. Same output
# (per edge, the indices of the faces containing it, in face order) from one pass
# over the faces with a hash map. Checked identical to the original at h/R = 1/12.
from jax_fdm.equilibrium.structures.meshes import Mesh as _JMesh


def _edges_faces_fast(self):
    lookup = {}
    for findex, face in enumerate(self.faces):
        loop = [int(v) for v in face if v >= 0]
        for u, v in zip(loop, loop[1:] + loop[:1]):
            lookup.setdefault(frozenset((u, v)), []).append(findex)
    return tuple(tuple(lookup.get(frozenset((int(u), int(v))), ()))
                 for u, v in self.edges)


_JMesh._edges_faces_original = _JMesh._edges_faces
_JMesh._edges_faces = _edges_faces_fast

# A2. JAXFDM_COMPACT=1: jax_fdm stores three mesh connectivity tables DENSE (edges x
# vertices, edges x faces, faces x vertices): ~12-15 GB at h/R = 1/48. The
# face-pressure load only ever reads, per edge, its two end nodes and its (at most
# two) adjacent faces, so the same quantities are carried as index lists and the two
# load routines that read the tables are replaced by gathers / scatter-adds of the
# same formulas (same edge order, same face order, same -1 fill). Off by default;
# checked against the dense path at h/R = 1/6-1/24: equilibria agree to 2e-14.
COMPACT = os.environ.get("JAXFDM_COMPACT", "0") == "1"
if COMPACT:
    from jax import vmap as _vmap
    from compas.numerical import connectivity_matrix as _cmat
    from jax.experimental.sparse import BCOO as _BCOO
    import jax_fdm.equilibrium.loads as _jl
    from jax_fdm.equilibrium.structures.meshes import MeshSparse as _JMeshSparse
    from jax_fdm.equilibrium.structures.graphs import GraphSparse as _JGraphSparse

    def _edges_faces_compact(self):
        ef = self._edges_faces()
        out = np.full((len(ef), 2), -1, dtype=np.int64)
        for i, fs in enumerate(ef):
            out[i, :len(fs)] = sorted(fs)[:2]     # flatnonzero order, -1 fill
        return jnp.asarray(out)

    def _faces_vertices_compact(self):
        return jnp.zeros((0,))                    # centroids from faces_indexed

    def _connectivity_compact(self):
        # forward-only use (equilibrium_state); the load path no longer reads it
        return _BCOO.from_scipy_sparse(_cmat(np.asarray(self.edges_indexed), "csc"))

    _JMeshSparse._connectivity_edges_faces_matrix = _edges_faces_compact
    _JMeshSparse._connectivity_faces_matrix = _faces_vertices_compact
    _JGraphSparse._connectivity_matrix = _connectivity_compact

    def _edges_tributary_faces_load(xyz, faces_load, structure):
        ends = jnp.sort(jnp.asarray(structure.edges_indexed), axis=1)
        fidx = structure.connectivity_edges_faces
        faces = jnp.asarray(structure.faces_indexed)
        ok = faces >= 0
        cent = (jnp.where(ok[..., None], xyz[jnp.where(ok, faces, 0)], 0.0).sum(1)
                / ok.sum(1)[:, None])

        def one(e, fi):
            areas = _vmap(_jl.edge_tributary_face_area, in_axes=(None, 0))(
                xyz[e, :], cent[fi, :])
            areas = jnp.where(fi >= 0, areas.ravel(), 0.0)
            return areas @ faces_load[fi, :]

        return _vmap(one)(ends, fidx)

    def _nodes_tributary_edges_load(edges_load, structure):
        e = jnp.asarray(structure.edges_indexed)
        n = int(np.asarray(structure.vertices).size)
        out = jnp.zeros((n, edges_load.shape[1]), edges_load.dtype)
        return 0.5 * out.at[e[:, 0]].add(edges_load).at[e[:, 1]].add(edges_load)

    _jl.edges_tributary_faces_load = _edges_tributary_faces_load
    _jl.nodes_tributary_edges_load = _nodes_tributary_edges_load


# =============================================================================
# Build the sparse structure + base parameters
# =============================================================================

vertices, faces_quad, faces_tri, fixed, APEX = build_tri()
# q = 1 on the A2 lattice, so N0 = sqrt(3) q and the analytic limit point is
# p* R / N0 = 2 exactly -- the same normalisation as the refinement table.
Q_SEED = float(os.environ.get("Q_EDGE_TRI", 1.0))

mesh = FDMesh.from_vertices_and_faces(vertices.tolist(), faces_quad)
for k in fixed:
    mesh.vertex_support(int(k))
for (u, v) in mesh.edges():
    mesh.edge_forcedensity((u, v), Q_SEED)
mesh.faces_loads([0.0, 0.0, 1.0])

structure = EquilibriumMeshStructureSparse.from_mesh(mesh)
bp = EquilibriumParametersState.from_datastructure(mesh)
QDTYPE = bp.q.dtype
UNIT_FACES = bp.loads.faces
FT = jnp.asarray(faces_tri)
N_EDGES = bp.q.shape[0]
N_V = vertices.shape[0]

FREE_NODES = np.asarray([v for v in range(N_V) if v not in set(fixed)])
FREE_ROWS = jnp.asarray(FREE_NODES)
IDX_FREE = np.asarray(structure.indices_free)
N_FREE = IDX_FREE.shape[0]
N_DOF = 3 * N_FREE

V0 = enclosed_volume_np(vertices, faces_tri)
THETA0 = jnp.asarray(np.log(np.full(N_EDGES, Q_SEED)), dtype=QDTYPE)

# Iteration cap for the damped Picard. This is a COMPILE-TIME bound: equinox's
# checkpointed while_loop builds O(log tmax) checkpoint machinery, so a needlessly
# large cap costs seconds of XLA compilation at every new gas_k. 4000 is ample --
# at the observed contraction factors the gas-law solve reaches eta=1e-14 in a few
# hundred iterations, and a prescribed-pressure solve that has not converged in
# 4000 is not going to (it has no equilibrium to find).
TMAX = int(os.environ.get("TMAX", 4000))


def zoned_theta0(gamma, beta, rb, w):
    """ln q per edge, in jax_fdm's own edge order, for the zoned cap of Sec. 3.1
    (fdm_limit_point.zones: soft disk gamma q0, belt beta q0, outer ring q0; zones of
    the flat reference). Returns (theta0, edge counts per zone)."""
    from fdm_limit_point.zones import zone_q
    E = list(mesh.edges())
    a = np.array([np.asarray(vertices[u_][:2], float) for u_, _ in E])
    b = np.array([np.asarray(vertices[v_][:2], float) for _, v_ in E])
    qe, ze = zone_q(a, b, rb, w, gamma, beta, Q_SEED)
    for (u_, v_), qv in zip(E, qe):
        mesh.edge_forcedensity((u_, v_), float(qv))
    th = jnp.asarray(np.log(np.asarray(EquilibriumParametersState.from_datastructure(mesh).q)),
                     dtype=QDTYPE)
    for (u_, v_) in mesh.edges():                       # restore the shared mesh object
        mesh.edge_forcedensity((u_, v_), Q_SEED)
    return th, np.bincount(ze, minlength=3)


def loss_from_xyz(xyz):
    """Smooth shape loss exercising every free node: mean squared height deviation."""
    return jnp.mean((xyz[FREE_ROWS, 2] - Z_TARGET) ** 2)


def free_of(xyz_full):
    return jnp.asarray(np.asarray(xyz_full)[IDX_FREE], dtype=QDTYPE)


# =============================================================================
# Inflating-sign detection
# =============================================================================

def _fixed_pressure_model(eta=1e-10, tmax=TMAX):
    return EquilibriumModelSparse(tmax=tmax, eta=eta, is_load_local=True,
                                  implicit_diff=True)


def solve_fixed_pressure(pressure, sign, eta=1e-10, tmax=TMAX):
    """Cold-start prescribed-pressure solve. Returns full-mesh xyz (may be non-finite)."""
    model = _fixed_pressure_model(eta, tmax)
    params = bp._replace(loads=bp.loads._replace(faces=UNIT_FACES * (sign * pressure)))
    return np.asarray(model(params, structure).xyz)


# jax_fdm's face normals follow the face winding; the sign that inflates the cap is
# found once by two cheap prescribed-pressure solves rather than assumed.
_zp = solve_fixed_pressure(0.5, +1.0)[APEX, 2]
_zm = solve_fixed_pressure(0.5, -1.0)[APEX, 2]
SIGN = 1.0 if (np.isfinite(_zp) and _zp >= _zm) else -1.0


# =============================================================================
# Damped Picard fixed-point solver (inside the differentiated fixed point)
# =============================================================================

DAMP_BETA = float(os.environ.get("DAMP_BETA", 0.5))


def make_damped_picard(beta=DAMP_BETA):
    """x <- (1-beta) x + beta f(x), stopping at the eta tolerance."""
    def solver_damped_picard(f, a, x_init, solver_config):
        tmax = solver_config["tmax"]
        eta = solver_config["eta"]

        def distance(x_prev, x):
            return jnp.mean(jnp.linalg.norm(x_prev - x, axis=1))

        def cond_fun(carry):
            x_prev, x = carry
            return distance(x_prev, x) > eta

        def body_fun(carry):
            _, x = carry
            return x, (1.0 - beta) * x + beta * f(a, x)

        x1 = (1.0 - beta) * x_init + beta * f(a, x_init)
        _, x_star = while_loop(cond_fun, body_fun, (x_init, x1),
                               max_steps=tmax, kind="checkpointed")
        return x_star

    return solver_damped_picard


DAMPED = make_damped_picard()


def make_mass_model(gas_k, eta):
    """The gas-law model of Sec. 2.2.1 at gas content K_gas = gas_k (Eq. 15)."""
    return ConstantMassModelSparse(
        tmax=TMAX, eta=eta, is_load_local=True, implicit_diff=True,
        itersolve_fn=solver_forward,
        faces_volume=FT, gas_k=gas_k, sign=SIGN, unit_faces=UNIT_FACES)


def mass_xyz(model, theta, xyz_free_init, eta):
    """Warm-started gas-law equilibrium, still wrapped by the implicit adjoint."""
    q = jnp.exp(theta).astype(QDTYPE)
    xyz_free = model.equilibrium_iterative_xyz(
        q, bp.xyz_fixed, bp.loads, structure,
        xyz_free_init=xyz_free_init, tmax=TMAX, eta=eta,
        solver=DAMPED, implicit_diff=True)
    return model.nodes_positions(xyz_free, bp.xyz_fixed, structure)


# =============================================================================
# Configurable adjoint backward rule (see module docstring)
# =============================================================================

def make_bwd(solver_obj, dense_stiffness=True):
    """Rebuild jax_fdm's adjoint rule with a chosen lineax solver and tolerance.

    dense_stiffness=False applies K^-1 with jax_fdm's own sparse solve (the one its
    sparse model uses in the forward pass) instead of a dense solve: same operator,
    needed at h/R = 1/48 where dense K is 0.5 GB and refactorised at every step."""
    from jax_fdm.equilibrium.sparse import sparse_solve

    def bwd(solver, solver_config, f, res, vec):
        theta, x_star = res
        p_fn = solver_config["loads_fn"]
        _, vjp_x = vjp(lambda x: p_fn(theta, x), x_star)

        K = theta[0]
        if dense_stiffness or not isinstance(K, JAXSparse):
            Kd = K.todense() if isinstance(K, JAXSparse) else K

            def linearsolve_fn(b):
                return jnp.linalg.solve(Kd, b)
        else:
            def linearsolve_fn(b):
                return sparse_solve(K, b)

        def A_fn(w):
            return w - vjp_x(linearsolve_fn(w))[0]

        op = FunctionLinearOperator(A_fn, jax.ShapeDtypeStruct(vec.shape, vec.dtype))
        w = linear_solve(op, vec, solver_obj, throw=False).value

        _, vjp_theta = vjp(lambda th: f(th, x_star), theta)
        return vjp_theta(w)[0], None

    return bwd


ADJOINT_RULES = {
    "shipped":  fp.fixed_point_bwd_adjoint,                        # NormalCG rtol=1e-6
    "normalcg": make_bwd(NormalCG(rtol=1e-12, atol=1e-14)),
    "gmres":    make_bwd(GMRES(rtol=1e-12, atol=1e-14, restart=64)),
    "bicgstab": make_bwd(BiCGStab(rtol=1e-12, atol=1e-14)),
    "gmres_sparse": make_bwd(GMRES(rtol=1e-12, atol=1e-14, restart=64),
                             dense_stiffness=False),
}


def use_adjoint(name):
    """Globally re-register the fixed-point backward rule. Affects every later grad."""
    fp.solver_fixedpoint_implicit.defvjp(fp.fixed_point_fwd, ADJOINT_RULES[name])
