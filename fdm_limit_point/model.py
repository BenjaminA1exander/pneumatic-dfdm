r"""model.py -- FDM operators, enclosed volume, and its first two derivatives.

This is the discrete problem of Sec. 2.1.2 of the paper (Eqs. 4-6).

SIGN CONVENTIONS (used everywhere in this package)
    Potential          Pi(x_u) = (1/2) sum_e q_e l_e^2  -  p V(x_u)
    Residual           r(x_u, p) = grad Pi
                                 = kron(K,I3) x_u + kron(D,I3) x_s - p grad V      (Eq. 4)
    Tangent            K_T = d r / d x_u = kron(K,I3) - p grad^2 V                 (Eq. 6)

    so r = 0 is equilibrium, and a state is stable when K_T is positive definite.
    The pressure acts along +grad V, i.e. it INFLATES: p > 0 raises V.

    Connectivity  C = [C_u  C_s]  (edges x nodes, +1 at the first node of an edge,
    -1 at the second), Q = diag(q_e), K = C_u^T Q C_u, D = C_u^T Q C_s. Coordinates
    are node-major (x0,y0,z0,x1,...), hence the kron(., I3): the force-density
    operator acts on nodes and is the identity on the three components.

    Volume  V = (1/6) sum_T x_a . (x_b x x_c) over anticlockwise triangles, origin
    at the rim centre in the plane z = 0. This is the divergence form (1/3) int x.n
    over the region closed by the flat base, whose own contribution vanishes because
    x . n = -z = 0 there. grad V and grad^2 V are taken by jax.grad / jax.hessian of
    that same functional, so the nodal pressure load is EXACTLY the gradient of the
    discrete volume and K_T is exactly symmetric -- the load is conservative by
    construction, not merely to within a lumping error.  (On a triangulated mesh the
    edge-to-centroid lumped face load of JAX-FDM is the same vector; see
    jaxfdm_gaslaw/common.py.)

FORCE DENSITIES
    q_mode="uniform"  q_e = q on every edge. Consistent on the equilateral interior
                      of the A2 lattice, where it represents tension N0 = sqrt(3) q
                      (Eq. 7).  This is the mode used throughout the paper.
    q_mode="cotan"    q_e = N0 w_e, w_e = (1/2) sum over incident triangles of the
                      cotangent of the opposite angle -- i.e. (1/2)(cot a + cot b)
                      on interior edges and (1/2) cot a on the rim edges. The weights
                      are taken on the mesh AS GENERATED (the flat lattice), which is
                      legitimate and non-circular: the cotangent Laplacian annihilates
                      affine functions at interior vertices, so for q_mode="cotan" the
                      generated flat mesh IS the p = 0 equilibrium exactly and there is
                      no circular dependence on the reference configuration. On an
                      equilateral triangle w_e = cot 60 = 1/sqrt(3), so N0 w_e = q and
                      the two modes agree in the interior by construction.  Kept as a
                      cross-check of Eq. 7; not used for any result in the paper.

Non-uniform force densities (the zoned cap of Sec. 3.1) are set by replacing K, D,
K3 and D3 of a "uniform" model; see sec31/cap_model.py.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Per-triangle volume kernels (jax): V_T = (1/6) x_a . (x_b x x_c)
# =============================================================================

def _vol_tri(p):
    a, b, c = p[0:3], p[3:6], p[6:9]
    return jnp.dot(a, jnp.cross(b, c)) / 6.0


_VOL = jax.jit(jax.vmap(_vol_tri))
_GRAD = jax.jit(jax.vmap(jax.grad(_vol_tri)))
_HESS = jax.jit(jax.vmap(jax.hessian(_vol_tri)))


def cotan_weights(pts, tris, n_nodes, edge_index, n_edges):
    """w_e = (1/2) sum over triangles containing e of cot(opposite angle)."""
    w = np.zeros(n_edges)
    for m in range(3):
        i = tris[:, m]
        j = tris[:, (m + 1) % 3]
        k = tris[:, (m + 2) % 3]
        u = pts[i] - pts[k]
        v = pts[j] - pts[k]
        cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        cot = np.einsum("ij,ij->i", u, v) / cross
        key = np.minimum(i, j) * n_nodes + np.maximum(i, j)
        np.add.at(w, edge_index[key], 0.5 * cot)
    return w


class Model:
    """Everything that depends on the mesh and on q, but not on the state."""

    def __init__(self, mesh, cfg, q_mode):
        self.mesh, self.cfg, self.q_mode = mesh, cfg, q_mode
        self.R, self.q, self.N0 = cfg.R, cfg.q, cfg.N0

        pts = mesh["xyz"]
        tris = mesh["tris"]
        edges = mesh["edges"]
        n = mesh["n_nodes"]
        self.free = mesh["free"]
        self.fixed = mesh["fixed"]
        self.n_free = len(self.free)
        self.ndof = 3 * self.n_free
        self.tris = tris
        self.edges = edges

        if q_mode == "uniform":
            self.q_edge = np.full(len(edges), self.q)
            self.w_min = float("nan")
        elif q_mode == "cotan":
            # dense n x n edge lookup: only the cotangent weights need it, and it
            # would be prohibitive on the finest meshes (n^2 int64 at h/R = 1/192)
            edge_index = np.full(n * n, -1, dtype=np.int64)
            edge_index[edges[:, 0] * n + edges[:, 1]] = np.arange(len(edges))
            w = cotan_weights(pts[:, :2], tris, n, edge_index, len(edges))
            self.q_edge = self.N0 * w
            self.w_min = float(w.min())
        else:
            raise ValueError("unknown q_mode " + repr(q_mode))

        # --- C = [C_u C_s], K, D ---------------------------------------------
        m = len(edges)
        rows = np.r_[np.arange(m), np.arange(m)]
        cols = np.r_[edges[:, 0], edges[:, 1]]
        vals = np.r_[np.ones(m), -np.ones(m)]
        C = sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()
        Q = sp.diags(self.q_edge)
        Cu, Cs = C[:, self.free], C[:, self.fixed]
        self.K = (Cu.T @ Q @ Cu).tocsc()
        self.D = (Cu.T @ Q @ Cs).tocsc()
        self.K3 = sp.kron(self.K, sp.eye(3), format="csc")
        self.D3 = sp.kron(self.D, sp.eye(3), format="csc")

        # --- sparsity pattern of grad^2 V restricted to the free dofs --------
        tri_dofs = (3 * tris[:, :, None] + np.arange(3)[None, None, :]
                    ).reshape(len(tris), 9)
        rr = np.repeat(tri_dofs, 9, axis=1).ravel()
        cc = np.tile(tri_dofs, (1, 9)).ravel()
        dofmap = np.full(3 * n, -1, dtype=np.int64)
        free_dofs = (3 * self.free[:, None] + np.arange(3)).ravel()
        dofmap[free_dofs] = np.arange(self.ndof)
        r_, c_ = dofmap[rr], dofmap[cc]
        keep = (r_ >= 0) & (c_ >= 0)
        self._h_rows, self._h_cols, self._h_keep = r_[keep], c_[keep], keep
        self._tri_dofs = tri_dofs
        self._free_dofs = free_dofs
        self._jtris = jnp.asarray(tris)

        # --- state vectors ----------------------------------------------------
        self.xs = pts[self.fixed].ravel().copy()          # supports, never move
        self.x_flat = None                                # set by reference()

    # -- geometry helpers ----------------------------------------------------
    def full_xyz(self, xu):
        X = np.empty((self.mesh["n_nodes"], 3))
        X[self.free] = xu.reshape(-1, 3)
        X[self.fixed] = self.xs.reshape(-1, 3)
        return X

    def _tri_coords(self, X):
        return jnp.asarray(X)[self._jtris].reshape(-1, 9)

    # -- volume and derivatives ----------------------------------------------
    def volume(self, xu):
        return float(np.sum(np.asarray(_VOL(self._tri_coords(self.full_xyz(xu))))))

    def grad_V(self, xu):
        g_tri = np.asarray(_GRAD(self._tri_coords(self.full_xyz(xu))))
        g = np.zeros(3 * self.mesh["n_nodes"])
        np.add.at(g, self._tri_dofs.ravel(), g_tri.ravel())
        return g[self._free_dofs]

    def hess_V(self, xu):
        h_tri = np.asarray(
            _HESS(self._tri_coords(self.full_xyz(xu)))).reshape(-1)
        return sp.coo_matrix(
            (h_tri[self._h_keep], (self._h_rows, self._h_cols)),
            shape=(self.ndof, self.ndof)).tocsc()

    # -- residual and tangent -------------------------------------------------
    def residual(self, xu, p, gV=None):
        """r = kron(K,I3) x_u + kron(D,I3) x_s - p grad V   (Eq. 4)."""
        gV = self.grad_V(xu) if gV is None else gV
        return self.K3 @ xu + self.D3 @ self.xs - p * gV

    def tangent(self, xu, p, H=None):
        """K_T = kron(K,I3) - p grad^2 V   (Eq. 6). Exactly symmetric by construction."""
        H = self.hess_V(xu) if H is None else H
        return (self.K3 - p * H).tocsc()

    # -- reference configuration ----------------------------------------------
    def reference(self):
        """Flat p = 0 equilibrium: K X_u = -D X_s with z = 0. Also builds the
        lumped mass m_i, one third of the reference triangle areas at node i."""
        lu = splu(self.K)
        Xs = self.xs.reshape(-1, 3)
        Xu = np.zeros((self.n_free, 3))
        for c in range(2):                        # z stays 0: K z = 0
            Xu[:, c] = lu.solve(-(self.D @ Xs[:, c]))
        self.x_flat = Xu.ravel().copy()

        Xref = self.full_xyz(self.x_flat)
        areas = 0.5 * np.linalg.norm(
            np.cross(Xref[self.tris[:, 1]] - Xref[self.tris[:, 0]],
                     Xref[self.tris[:, 2]] - Xref[self.tris[:, 0]]), axis=1)
        mass_node = np.zeros(self.mesh["n_nodes"])
        for m in range(3):
            np.add.at(mass_node, self.tris[:, m], areas / 3.0)
        self.area_ref = float(areas.sum())
        self.mass_node = mass_node
        self.m_free = np.repeat(mass_node[self.free], 3)
        self.M3 = sp.diags(self.m_free, format="csc")
        # how far the generated lattice sits from its own flat equilibrium
        self.flat_drift = float(np.abs(
            self.x_flat - self.mesh["xyz"][self.free].ravel()).max())
        return self.x_flat

    # -- nodal normals of the deformed state ----------------------------------
    def nodal_normals(self, xu):
        X = self.full_xyz(xu)
        a, b, c = X[self.tris[:, 0]], X[self.tris[:, 1]], X[self.tris[:, 2]]
        an = 0.5 * np.cross(b - a, c - a)          # area-weighted, outward
        nn = np.zeros((self.mesh["n_nodes"], 3))
        for m in range(3):
            np.add.at(nn, self.tris[:, m], an)
        nrm = np.linalg.norm(nn, axis=1, keepdims=True)
        return nn / np.maximum(nrm, 1e-300)
