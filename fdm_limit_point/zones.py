r"""zones.py -- force-density zones of the "zoned" cap of Sec. 3.1 (local limit point).

The zoned cap has three zones in the flat reference disk (radii in units of R):

    soft disk    r <  RB - W/2      q = GAMMA q0   (GAMMA = 0.25: inflates into a bulb)
    belt         |r - RB| <= W/2    q = BETA  q0   (BETA = 10: the stiff ring, the neck)
    outer ring   r >  RB + W/2      q = q0

A soft disk held by a stiff belt bulges into a bulb that reaches its own hemisphere,
and with it a pressure maximum, long before the cap as a whole does.  That is the
"local limit point" of the paper (Fig. 3, zoned cap).  GAMMA = BETA = 1 recovers the
uniform cap of Sec. 2.1.2 (global limit point).

Two ways of assigning q to the edges that cross a zone boundary:

    midpoint  (BLEND=0)  each edge takes the zone of its reference midpoint
    blend     (BLEND=1)  each edge takes the length-weighted mean of the zones it
                         crosses, q_e = sum_k (fraction of the edge in zone k) q_k,
                         so the belt width does not jump with the mesh (default; all
                         results in the paper use this scheme).
"""
import os

import numpy as np

BLEND = os.environ.get("BLEND", "1") == "1"


def edge_fractions(a, b, rb, w):
    """Fractions of the straight segments a -> b (arrays (n, 2)) lying in the soft disk,
    the belt and the outer ring, in that order.  Each segment is cut at its
    intersections with the two circles r = rb -/+ w/2 and every piece is assigned by
    the radius of its own midpoint."""
    r1, r2 = rb - w / 2, rb + w / 2
    d = b - a
    A = np.einsum("ij,ij->i", d, d)
    Bq = 2 * np.einsum("ij,ij->i", a, d)
    C0 = np.einsum("ij,ij->i", a, a)
    out = np.zeros((len(a), 3))
    for i in range(len(a)):
        ts = [0.0, 1.0]
        for rr in (r1, r2):                       # |a + t d| = rr  (quadratic in t)
            disc = Bq[i] ** 2 - 4 * A[i] * (C0[i] - rr ** 2)
            if disc > 0:
                for s in (-1, 1):
                    t = (-Bq[i] + s * np.sqrt(disc)) / (2 * A[i])
                    if 0 < t < 1:
                        ts.append(t)
        ts = np.sort(ts)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            rm = np.linalg.norm(a[i] + 0.5 * (t0 + t1) * d[i])
            z = 0 if rm < r1 else (1 if rm <= r2 else 2)
            out[i, z] += t1 - t0
    return out


def zone_q(a, b, rb, w, gamma, beta, q0=1.0):
    """(q per edge, majority zone per edge) for the edges a -> b of the flat reference."""
    qz = np.array([gamma, beta, 1.0]) * q0
    if BLEND:
        fr = edge_fractions(a, b, rb, w)
        return fr @ qz, np.argmax(fr, axis=1)
    rm = np.linalg.norm(0.5 * (a + b), axis=1)
    z = np.where(rm < rb - w / 2, 0, np.where(rm <= rb + w / 2, 1, 2))
    return qz[z], z
