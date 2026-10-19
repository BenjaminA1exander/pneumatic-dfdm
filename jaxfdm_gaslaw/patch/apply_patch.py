r"""apply_patch.py -- install (or check) the gas-law patch of jax_fdm 0.10.0.

The differentiable force density method used in the paper is JAX-FDM
(https://github.com/arpastrana/jax_fdm, MIT licence), release 0.10.0 from PyPI.
Three of its files are replaced by the versions in this folder:

    equilibrium/loads.py               face_xyz fix: a padded triangular face of a
                                       mixed tri/quad mesh no longer wraps to the
                                       last vertex (bug fix, no effect on the all-
                                       triangle cap meshes of Sec. 3.1)
    equilibrium/models.py              `global_load_fn` hook on EquilibriumModel and
                                       ConstantMassModelSparse: the gas-law load
                                       p = K_gas / V(x) of Eq. 16, differentiated by
                                       the implicit adjoint (Sec. 2.2.1)
    equilibrium/solvers/fixed_point.py host-side tally of forward / adjoint solves
                                       that stop without converging (exit summary)

jax_fdm-0.10.0-gaslaw.diff is the unified diff against the PyPI wheel, and the two
sha256 files identify the pristine and the patched versions of each file.

Usage (in the environment that has jax_fdm installed):
    python apply_patch.py --check       report the state of each file, exit 1 if not patched
    python apply_patch.py               apply (originals are kept as <file>.orig)
    python apply_patch.py --restore     put the originals back
"""
import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "loads.py": os.path.join("equilibrium", "loads.py"),
    "models.py": os.path.join("equilibrium", "models.py"),
    "fixed_point.py": os.path.join("equilibrium", "solvers", "fixed_point.py"),
}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_hashes(name):
    out = {}
    with open(os.path.join(HERE, name)) as fh:
        for line in fh:
            h, _, p = line.strip().partition(" ")
            out[os.path.basename(p.lstrip("*"))] = h
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--restore", action="store_true", help="restore the .orig files")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that match neither the pristine nor the patched hash")
    args = ap.parse_args(argv)

    import jax_fdm
    pkg = os.path.dirname(os.path.abspath(jax_fdm.__file__))
    version = getattr(jax_fdm, "__version__", "?")
    print(f"jax_fdm {version} at {pkg}")
    if version != "0.10.0":
        print("WARNING: the patch was written against jax_fdm 0.10.0")

    pristine = read_hashes("sha256_pypi_0.10.0.txt")
    patched = read_hashes("sha256_patched.txt")
    states = {}
    for name, rel in FILES.items():
        target = os.path.join(pkg, rel)
        h = sha256(target)
        states[name] = ("patched" if h == patched[name] else
                        "pristine" if h == pristine[name] else "unknown")
        print(f"  {rel:45s} {states[name]}")

    if args.check:
        return 0 if all(s == "patched" for s in states.values()) else 1

    if args.restore:
        for name, rel in FILES.items():
            target = os.path.join(pkg, rel)
            if os.path.exists(target + ".orig"):
                shutil.copyfile(target + ".orig", target)
                print(f"  restored {rel}")
            else:
                print(f"  no backup for {rel}; nothing done")
        return 0

    if all(s == "patched" for s in states.values()):
        print("already patched; nothing to do")
        return 0
    if any(s == "unknown" for s in states.values()) and not args.force:
        print("a target file matches neither hash (local modifications?); use --force to overwrite")
        return 2
    for name, rel in FILES.items():
        target = os.path.join(pkg, rel)
        if states[name] == "patched":
            continue
        if not os.path.exists(target + ".orig"):
            shutil.copyfile(target, target + ".orig")
        shutil.copyfile(os.path.join(HERE, name), target)
        print(f"  patched {rel}")
    # byte-compiled caches of the old files must not shadow the new sources
    for root, _dirs, files in os.walk(pkg):
        for f in files:
            if f.endswith(".pyc"):
                os.remove(os.path.join(root, f))
    print("done; run with --check to confirm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
