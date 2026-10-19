r"""mesh.py -- equilateral triangular lattice clipped to the disk of radius R.

GEOMETRY
    Lattice points sit at  (i + j/2) h,  j (sqrt(3)/2) h  for integer i, j, which
    is the A2 lattice: every interior vertex has exactly six neighbours at 60
    degrees, so every interior triangle is equilateral. That is the configuration
    in which uniform force densities q_e = q are a consistent discretisation of a
    membrane of tension N0 = sqrt(3) q (equivalently, cot 60 = 1/sqrt(3), so the
    cotangent weight of every interior edge is 1/sqrt(3) and N0 w_e = q).

    Points with r > R - trim*h are removed so that no lattice point lands
    arbitrarily close to the rim; n_rim = round(2 pi R / h) rim nodes are then
    placed exactly on the circle. scipy.spatial.Delaunay triangulates the union.
    The point set is convex-hulled by the rim (every retained lattice point is
    strictly inside), so the triangulation covers the disk with no holes and no
    rim-only triangles: three cocircular rim nodes have the rim circle itself as
    circumcircle, which contains every interior point, so Delaunay never keeps
    such a triangle.

ORIENTATION
    Every triangle is stored anticlockwise seen from +z in the flat state. That
    fixes the sign of V = (1/6) sum_T x_a . (x_b x x_c) to be POSITIVE for a cap
    bulging towards +z: the triangle normals are then the outward normals of the
    closed region bounded by the cap and the disk z = 0, whose base contributes
    nothing to the divergence form because x . n = -z = 0 there.

SUPPORTS
    The n_rim rim nodes are fixed; every lattice node is free.
"""
import math

import numpy as np
from scipy.spatial import Delaunay


def lattice_disk(h, radius, trim):
    """A2 lattice points with r <= radius - trim*h, origin included."""
    n = int(math.ceil(2.0 * radius / h)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    x = (i + 0.5 * j) * h
    y = j * (0.5 * math.sqrt(3.0) * h)
    keep = np.hypot(x, y) <= radius - trim * h + 1e-12
    return np.column_stack([x[keep], y[keep]])


def rim_nodes(h, radius):
    """n_rim = round(2 pi R / h) nodes placed exactly on the circle."""
    n_rim = int(round(2.0 * math.pi * radius / h))
    th = 2.0 * math.pi * np.arange(n_rim) / n_rim
    return np.column_stack([radius * np.cos(th), radius * np.sin(th)])


def orient_ccw(pts, tris):
    """Force anticlockwise winding in the xy plane (seen from +z)."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    twice_area = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    flip = twice_area < 0.0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def triangle_angles(pts, tris):
    """All three angles of every triangle, in degrees."""
    p = pts[tris]                                     # (nt, 3, 2 or 3)
    out = np.empty((len(tris), 3))
    for m in range(3):
        o, a, b = p[:, m], p[:, (m + 1) % 3], p[:, (m + 2) % 3]
        u, v = a - o, b - o
        cs = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        out[:, m] = np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))
    return out


def edges_of(tris):
    """Unique undirected edges, sorted -- the FDM edge list."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def triangle_areas(pts3, tris):
    a, b, c = pts3[tris[:, 0]], pts3[tris[:, 1]], pts3[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def build(level, cfg):
    """Mesh for h/R = 1/level. Walks cfg.trim_ladder until the minimum angle
    clears cfg.min_angle_target; the trim actually used is reported."""
    h = cfg.R / level
    attempts = []
    ladder = list(cfg.trim_ladder)
    if cfg.trim not in ladder:
        ladder.insert(0, cfg.trim)
    for trim in ladder:
        lat = lattice_disk(h, cfg.R, trim)
        rim = rim_nodes(h, cfg.R)
        pts = np.vstack([lat, rim])
        tris = orient_ccw(pts, Delaunay(pts).simplices.astype(int))
        ang = triangle_angles(pts, tris)
        attempts.append((trim, float(ang.min())))
        if ang.min() >= cfg.min_angle_target:
            break

    n_lat = len(lat)
    n_rim = len(rim)
    xyz = np.zeros((len(pts), 3))
    xyz[:, :2] = pts
    fixed = np.arange(n_lat, n_lat + n_rim)
    free = np.arange(n_lat)

    # The origin is always an A2 lattice point: it is the apex node.
    apex = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))

    return dict(
        level=int(level), h=float(h), trim=float(trim),
        xyz=xyz, tris=tris, edges=edges_of(tris),
        free=free, fixed=fixed, apex=apex,
        n_nodes=len(pts), n_free=int(n_lat), n_rim=int(n_rim),
        min_angle=float(ang.min()), max_angle=float(ang.max()),
        trim_attempts=attempts,
    )
