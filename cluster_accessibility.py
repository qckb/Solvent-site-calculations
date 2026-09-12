#!/usr/bin/env python3
"""
cluster_accessibility.py  --  metal-site steric accessibility, isolated cluster
vs a 2D hexagonal arrangement (clusters on the hexagon corners AND the centre,
i.e. the triangular Bravais lattice with nearest-neighbour distance a).

METRIC
  Omega/4pi : fraction of directions around a metal atom along which a spherical
              probe of radius r_probe can travel in from infinity to touch that
              atom without overlapping any other atom's van der Waals sphere.
  SASA      : Shrake-Rupley solvent-accessible surface area of the metal atom.

  "Retention" = (value in the hexagonal sheet) / (value in the isolated cluster).

REQUIREMENTS
  Python 3.8+ and numpy.  Nothing else.  No internet, no ASE, no compilation.
      pip install numpy

QUICK START
  python cluster_accessibility.py ag44_sr30.xyz --metal Ag
  python cluster_accessibility.py ag44_sr30.xyz --metal Ag --probe 1.65 --scan
  python cluster_accessibility.py ag44_sr30.xyz --metal Ag --a 22 25 28 --peratom
  python cluster_accessibility.py ag44_sr30.xyz --metal Ag --csv results.csv
  python cluster_accessibility.py ag44_sr30.xyz --metal Ag --3d --a 30.5 --c 30.5
  python cluster_accessibility.py ag44_sr30.xyz --metal Ag --3d --stacking ABCABC

INPUT
  .xyz (plain or extended), or POSCAR/CONTCAR.  Coordinates in Angstrom.
  The cluster is recentred automatically; the sheet is built in the xy plane
  unless you pass --normal.
"""

import argparse
import sys

import numpy as np

# ---------------------------------------------------------------- vdW radii (A)
VDW = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Sc": 2.11, "Ti": 2.00, "V": 2.00, "Cr": 2.00,
    "Mn": 2.00, "Fe": 2.00, "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
    "Rb": 3.03, "Sr": 2.49, "Y": 2.19, "Zr": 2.06, "Nb": 1.98, "Mo": 1.90,
    "Tc": 1.83, "Ru": 1.78, "Rh": 1.73, "Pd": 1.63, "Ag": 1.72, "Cd": 1.58,
    "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16,
    "Cs": 3.43, "Ba": 2.68, "La": 2.40, "Hf": 2.06, "Ta": 2.00, "W": 2.10,
    "Re": 2.05, "Os": 2.00, "Ir": 2.00, "Pt": 1.75, "Au": 1.66, "Hg": 1.55,
    "Tl": 1.96, "Pb": 2.02, "Bi": 2.07,
}

# Handy probe radii, printed by --help-probes
PROBES = {"H2O": 1.40, "H2": 1.20, "O2": 1.50, "CO2": 1.65,
          "CH4": 1.90, "benzene": 2.60}


# ------------------------------------------------------------------ structure IO
def read_xyz(path):
    txt = open(path).read().replace("\r\n", "\n").replace("\r", "\n")
    lines = txt.split("\n")
    nat = int(lines[0].split()[0])
    sym, pos = [], []
    for ln in lines[2:2 + nat]:
        f = ln.split()
        sym.append(f[0].capitalize())
        pos.append([float(x) for x in f[1:4]])
    return sym, np.array(pos)


def read_poscar(path):
    L = open(path).read().replace("\r\n", "\n").split("\n")
    scale = float(L[1].split()[0])
    cell = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]) * scale
    names, counts = L[5].split(), [int(x) for x in L[6].split()]
    sym = [s.capitalize() for s, n in zip(names, counts) for _ in range(n)]
    i = 7
    if L[i].strip()[:1].lower() == "s":
        i += 1
    direct = L[i].strip()[:1].lower() == "d"
    i += 1
    raw = np.array([[float(x) for x in L[i + k].split()[:3]] for k in range(sum(counts))])
    return sym, (raw @ cell if direct else raw * scale)


def read_structure(path):
    head = path.rsplit("/", 1)[-1].upper()
    if "POSCAR" in head or "CONTCAR" in head or path.lower().endswith(".vasp"):
        return read_poscar(path)
    return read_xyz(path)


# --------------------------------------------------------------------- geometry
def fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def omega_free(pos, radii, im, probe, u, rcut=22.0, chunk=400):
    """Boolean mask over directions u that are open for atom im."""
    d = pos - pos[im]
    keep = (d * d).sum(axis=1) < rcut ** 2
    keep[im] = False
    c, R = d[keep], radii[keep] + probe
    t0 = radii[im] + probe                     # probe centre at contact
    blocked = np.zeros(len(u), dtype=bool)
    for s in range(0, len(c), chunk):
        cc, RR = c[s:s + chunk], R[s:s + chunk]
        proj = u @ cc.T
        d2 = (cc * cc).sum(axis=1)[None, :]
        perp2 = np.maximum(d2 - proj ** 2, 0.0)
        # closest approach of the probe centre for path parameter t >= t0
        mind2 = np.where(proj >= t0, perp2, d2 - 2.0 * t0 * proj + t0 ** 2)
        blocked |= (mind2 < (RR ** 2)[None, :]).any(axis=1)
        if blocked.all():
            break
    return ~blocked


def sasa_atom(pos, radii, im, probe, npts=4802, rcut=22.0):
    """Shrake-Rupley SASA of a single atom, in A^2."""
    Ri = radii[im] + probe
    sphere = fibonacci_sphere(npts) * Ri + pos[im]
    d = pos - pos[im]
    keep = (d * d).sum(axis=1) < rcut ** 2
    keep[im] = False
    if not keep.any():
        return 4.0 * np.pi * Ri ** 2
    c, Rj = pos[keep], radii[keep] + probe
    buried = np.zeros(npts, dtype=bool)
    for s in range(0, len(c), 200):
        dist2 = ((sphere[:, None, :] - c[None, s:s + 200, :]) ** 2).sum(axis=2)
        buried |= (dist2 < (Rj[s:s + 200] ** 2)[None, :]).any(axis=1)
    return (~buried).mean() * 4.0 * np.pi * Ri ** 2


def hex_lattice(pos, a, nrep=5, normal=np.array([0.0, 0.0, 1.0])):
    """Clusters on hexagon corners + centre = triangular lattice, spacing a.
    The central image is emitted first so original atom indices stay valid."""
    n = normal / np.linalg.norm(normal)
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ n) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - n * (e1 @ n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    v1, v2 = a * e1, a * (0.5 * e1 + np.sqrt(3) / 2 * e2)
    half = nrep // 2
    sh = [i * v1 + j * v2 for i in range(-half, nrep - half)
          for j in range(-half, nrep - half)]
    sh.sort(key=np.linalg.norm)
    return np.vstack([pos + s for s in sh]), len(sh)


def lattice_3d(pos, a, c, nxy=3, nz=1, stacking="ABAB", normal=np.array([0.0, 0.0, 1.0])):
    """3D hexagonal superlattice: triangular layers of spacing a, stacked along
    `normal` with interlayer distance c.  stacking = 'AAA' (simple hexagonal),
    'ABAB' (hcp-like, alternate layers over the hollow site) or 'ABCABC' (fcc-like).
    Central image first, so original atom indices stay valid."""
    n = normal / np.linalg.norm(normal)
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ n) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - n * (e1 @ n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    v1, v2 = a * e1, a * (0.5 * e1 + np.sqrt(3) / 2 * e2)
    hollow = (v1 + v2) / 3.0
    nper = {"AAA": 1, "ABAB": 2, "ABCABC": 3}[stacking.upper()]
    sh = []
    for k in range(-nz, nz + 1):
        off = hollow * (k % nper)
        for i in range(-nxy, nxy + 1):
            for j in range(-nxy, nxy + 1):
                sh.append(i * v1 + j * v2 + c * k * n + off)
    sh.sort(key=np.linalg.norm)
    return np.vstack([pos + s for s in sh]), len(sh)


def min_interatomic(pos, P):
    """Closest approach between the central cluster and all its images."""
    nb = P[len(pos):]
    if len(nb) == 0:
        return np.inf
    best = np.inf
    for s in range(0, len(nb), 2000):
        d2 = ((pos[:, None, :] - nb[None, s:s + 2000, :]) ** 2).sum(-1)
        best = min(best, float(np.sqrt(d2.min())))
    return best


def contact_spacing(pos, radii, normal):
    """Nearest-neighbour distance at which two rigid clusters just touch."""
    n = normal / np.linalg.norm(normal)
    inplane = pos - np.outer(pos @ n, n)
    return 2.0 * (np.linalg.norm(inplane, axis=1) + radii).max()


# ------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(
        description="Metal accessibility: isolated cluster vs hexagonal sheet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="probe radii (A):  " + "  ".join(f"{k}={v}" for k, v in PROBES.items()))
    p.add_argument("structure", help=".xyz / extended .xyz / POSCAR")
    p.add_argument("--metal", default=None, help="element symbol, e.g. Ag")
    p.add_argument("--index", type=int, nargs="+", default=None,
                   help="0-based atom indices instead of --metal")
    p.add_argument("--probe", type=float, default=1.4, help="probe radius A (default 1.4)")
    p.add_argument("--a", type=float, nargs="+", default=None,
                   help="lattice spacings A; default = contact and contact-8..contact")
    p.add_argument("--scan", action="store_true",
                   help="also sweep probe = 1.2 1.4 1.65 2.0")
    p.add_argument("--nrep", type=int, default=5, help="in-plane replication (default 5x5)")
    p.add_argument("--3d", dest="d3", action="store_true",
                   help="3D superlattice instead of a single layer")
    p.add_argument("--c", type=float, nargs="+", default=None,
                   help="interlayer spacings A for --3d (default: same as a)")
    p.add_argument("--stacking", default="ABAB", choices=["AAA", "ABAB", "ABCABC"],
                   help="layer stacking for --3d (default ABAB)")
    p.add_argument("--nz", type=int, default=1, help="layers above and below (default 1)")
    p.add_argument("--npts", type=int, default=20000, help="ray directions")
    p.add_argument("--normal", type=float, nargs=3, default=[0, 0, 1],
                   help="sheet normal (default 0 0 1)")
    p.add_argument("--peratom", action="store_true", help="per-atom table + SASA")
    p.add_argument("--csv", default=None, help="write results to this CSV")
    args = p.parse_args()

    sym, pos = read_structure(args.structure)
    pos = pos - pos.mean(axis=0)
    radii = np.array([VDW.get(s, 2.0) for s in sym])
    nrm = np.array(args.normal, dtype=float)

    if args.index is not None:
        sites = np.array(args.index)
    elif args.metal is not None:
        sites = np.array([i for i, s in enumerate(sym) if s == args.metal.capitalize()])
    else:
        sys.exit("give --metal SYMBOL or --index N [N ...]")
    if len(sites) == 0:
        sys.exit(f"no {args.metal} atoms found in {args.structure}")

    u = fibonacci_sphere(args.npts)
    ac = contact_spacing(pos, radii, nrm)
    spacings = args.a if args.a else [ac - 8, ac - 5, ac - 2, ac]

    print(f"structure   : {args.structure}  ({len(sym)} atoms)")
    print(f"sites       : {len(sites)} x {args.metal or 'index'}")
    print(f"rigid-sphere contact spacing a_contact = {ac:.1f} A")
    print(f"sheet       : {args.nrep}x{args.nrep} images, normal {nrm}\n")

    probes = [1.2, 1.4, 1.65, 2.0] if args.scan else [args.probe]
    rows = []
    for probe in probes:
        om0 = np.array([omega_free(pos, radii, i, probe, u).mean() for i in sites])
        print(f"probe {probe:4.2f} A  |  isolated cluster: sum(Omega/4pi) = {om0.sum():.4f}, "
              f"{(om0 > 1e-3).sum()} of {len(sites)} sites exposed")
        if om0.sum() == 0:
            print("   site is fully buried even as an isolated cluster.\n")
            continue
        # monotonicity: adding images never opens a direction, so zeros stay zero
        live = sites[om0 > 0]
        hdr = f"   {'a (A)':>8} {'c (A)':>8} " if args.d3 else f"   {'a (A)':>8} "
        print(hdr + f"{'min d':>8} {'sum Omega':>10} {'retention':>10}   status")
        clist = (args.c if args.c else spacings) if args.d3 else [None]
        for a in spacings:
            for c in (clist if args.d3 else [None]):
                if args.d3:
                    P, ni = lattice_3d(pos, a, c, args.nrep // 2 + 1, args.nz,
                                       args.stacking, nrm)
                else:
                    P, ni = hex_lattice(pos, a, args.nrep, nrm)
                RR = np.tile(radii, ni)
                md = min_interatomic(pos, P)
                om = sum(omega_free(P, RR, i, probe, u).mean() for i in live)
                ret = om / om0.sum()
                status = "OK" if md > 2.5 else "REJECT: atoms overlap"
                pre = f"   {a:8.1f} {c:8.1f} " if args.d3 else f"   {a:8.1f} "
                print(pre + f"{md:8.2f} {om:10.4f} {ret:10.3f}   {status}")
                rows.append(dict(probe=probe, a=a, c=c if c else 0.0, min_d=md,
                                 sum_omega=om, retention=ret, ok=md > 2.5))
        print()

    if args.peratom:
        probe = args.probe
        print(f"per-atom, isolated cluster, probe {probe:.2f} A "
              f"(showing exposed sites only)")
        print(f"   {'idx':>5} {'elem':>5} {'r_centroid':>11} {'Omega/4pi':>10} {'SASA':>8}")
        tot = 0.0
        for i in sites:
            om = omega_free(pos, radii, i, probe, u).mean()
            sa_ = sasa_atom(pos, radii, i, probe)
            tot += sa_
            if om > 1e-4 or sa_ > 0.05:
                print(f"   {i:5d} {sym[i]:>5} {np.linalg.norm(pos[i]):11.2f} "
                      f"{om:10.4f} {sa_:8.2f}")
        print(f"   total site SASA = {tot:.2f} A^2\n")

    if args.csv and rows:
        with open(args.csv, "w") as fh:
            fh.write("probe,a,c,min_d,sum_omega,retention,physically_valid\n")
            for r in rows:
                fh.write(f"{r['probe']},{r['a']:.2f},{r['c']:.2f},{r['min_d']:.2f},"
                         f"{r['sum_omega']:.6f},{r['retention']:.6f},{int(r['ok'])}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
