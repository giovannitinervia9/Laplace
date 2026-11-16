"""Reference values for factor_analytic_pdmat at small specific variances
(psi_i = e^(2 t_i) from e^-2 to e^-8), where M = Lambda Lambda^T + Psi is close
to rank d. Same method, format and quantities as pdmat_factor_analytic.py
(mpmath at 640 bits from the definition of M, mixed central differences with
h = 2^-60 checked against h = 2^-70), whose functions it uses.

Usage (from the repository root):
    python -I tools/reference/pdmat_factor_analytic_small_psi.py [shard nshards]
File: tests/data/reference/pdmat_factor_analytic_small_psi.csv (with shards,
one file per shard, .<shard>.csv, joined by the last call without arguments
when all shard files exist).

Points, for (p, d) in {(3, 1), (4, 1), (4, 2), (5, 2), (3, 3), (4, 4)}:
- loadings as the spread point of pdmat_factor_analytic.py (off-diagonal
  +-1.5 alternating, diagonal 0.4), every t_i = -1, -2, -3, -4
  (psi = e^-2 ... e^-8);
- the same loadings with t_i = -3 on even rows and 1 on odd rows;
- random loadings N(0, 0.5^2) (random.Random(20261116 + 10 p + d)) with every
  t_i = -3;
and the hard point of tools/reference/distrib_mvgaussian.py at (4, 1):
Lambda = (e^0.4, 1.5, -1.5, 1.5), t = (-3, -3, 1, -3).
Tuples: as pdmat_factor_analytic.py (its tuples(), seeded per point).
"""
import random
import sys
import time
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdmat_factor_analytic as base  # noqa: E402

OUT = base.OUT.with_name("pdmat_factor_analytic_small_psi.csv")
SHAPES = [(3, 1), (4, 1), (4, 2), (5, 2), (3, 3), (4, 4)]


def points():
    out = []
    for p, d in SHAPES:
        lay = base.layout(p, d)
        spread = base.points(p, d)[1]
        nl = len(lay) - p
        for t in (-1.0, -2.0, -3.0, -4.0):
            out.append((p, d, spread[:nl] + [t] * p))
        out.append((p, d, spread[:nl] + [-3.0 if i % 2 == 0 else 1.0 for i in range(p)]))
        rng = random.Random(20261116 + 10 * p + d)
        out.append((p, d, [rng.gauss(0.0, 0.5) for _ in range(nl)] + [-3.0] * p))
    out.append((4, 1, [0.4, 1.5, -1.5, 1.5, -3.0, -3.0, 1.0, -3.0]))
    return out


def run(pts):
    rows = []
    worst = mp.mpf(0)
    for n, (p, d, tf) in enumerate(pts):
        t0 = time.time()
        theta = [mp.mpf(x) for x in tf]
        tstr = ";".join(float.hex(x) for x in tf)
        c1, c2 = {}, {}
        rng = random.Random(7 + 1000 * n)
        for tup in base.tuples(p, d, rng):
            d1 = base.derivative(p, d, theta, tup, base.H, c1)
            d2 = base.derivative(p, d, theta, tup, base.H_CHECK, c2)
            for q in range(3):
                scale = 1 + max(abs(x) for x in d1[q].values())
                for key in sorted(d1[q]):
                    r = abs(d1[q][key] - d2[q][key]) / scale
                    if r > base.TOL:
                        raise SystemExit("h-check failed: point %r tuple %r q %d key %r: %s"
                                         % (tf, tup, q, key, mp.nstr(r, 5)))
                    worst = max(worst, r)
                    hi, lo = base.hex_pair(d1[q][key])
                    rows.append("%d,%d,%s,%d,%d,%s,%d,%d,%s,%s" % (
                        p, d, tstr, q, len(tup), ";".join(map(str, tup)) if tup else "-",
                        key[0], key[1], hi, lo))
        print("point %d (p %d d %d): %d rows, %.1f s" % (n, p, d, len(rows), time.time() - t0))
        sys.stdout.flush()
    return rows, worst


def main():
    mp.mp.prec = base.PREC
    pts = points()
    if len(sys.argv) == 3:
        shard, nsh = int(sys.argv[1]), int(sys.argv[2])
        rows, worst = run([(p, d, t) for k, (p, d, t) in enumerate(pts) if k % nsh == shard])
        path = OUT.with_name(OUT.stem + ".%d.csv" % shard)
        path.write_text("".join(r + "\n" for r in rows), encoding="utf-8", newline="\n")
        print("shard %d: %d rows, worst h disagreement %s" % (shard, len(rows), mp.nstr(worst, 5)))
        return
    shards = sorted(OUT.parent.glob(OUT.stem + ".*.csv"))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(base.HEADER + "\n")
        for s in shards:
            f.write(s.read_text(encoding="utf-8"))
    for s in shards:
        s.unlink()
    print("joined %d shards into %s" % (len(shards), OUT))


if __name__ == "__main__":
    main()
