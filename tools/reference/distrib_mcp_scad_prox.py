"""Reference values for the prox in y of mcp_distrib and scad_distrib, over the three
properizing tails (laplace::distrib; docs/numerics.md, "The prox of the MCP and SCAD laws").

The prox is the global minimizer of

    g(x) = t q(x) + (x - U)^2 / 2,   x >= 0,   U = |v - mu|,

with q the law's properized penalty. The reference comes from the definition, not from
the closed candidates of the implementation: at 200 bits, g' is formed from q' and
bisected on every sign change of a dense scan of [0, U + knot], and the stationary
points so found are compared with the ends on g itself. The row carries the minimizer
and the gap to the best other candidate, so that a test can tell a rounding tie, where
the global minimizer jumps between branches, from an error.

Writes tests/data/reference/distrib_mcp_scad_prox.csv with the columns

    law,tail,lam,p,t1,t2,U,t,hi,lo,gap

where law is 0 for mcp and 1 for scad, tail is 0 gaussian, 1 student, 2 logistic,
p is gamma for mcp and a for scad, (t1, t2) are the tail's parameters (t2 unused
outside the Student tail), hi and lo are the minimizer as two hex doubles, and gap
is the objective's distance to the runner-up, relative to max(1, |g|).

Run with python -I.
"""
import os

from mpmath import mp, mpf, sqrt, log, cosh, tanh, log1p, exp

mp.prec = 200

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "tests", "data", "reference", "distrib_mcp_scad_prox.csv")


def hex_pair(x):
    """x as hi + lo, two doubles in C99 hex, hi the nearest double and lo the remainder."""
    hi = float(x)
    lo = float(x - mpf(hi))
    return hi.hex(), lo.hex()


# ---------------------------------------------------------------- the three tails
# H(z) = kappa Psi(z / varsigma), with H(0) = H'(0) = 0


class GaussianTail:
    code, n_par = 0, 1

    def __init__(self, V):
        self.par = (V, mpf(1))

    def H(self, z):
        return z * z / (2 * self.par[0])

    def dH(self, z):
        return z / self.par[0]


class StudentTail:
    code, n_par = 1, 2

    def __init__(self, s, nu):
        self.par = (s, nu)

    def varsigma(self):
        return self.par[0] * sqrt(self.par[1])

    def H(self, z):
        return (self.par[1] + 1) / 2 * log1p((z / self.varsigma()) ** 2)

    def dH(self, z):
        v2 = self.varsigma() ** 2
        return (self.par[1] + 1) * z / (v2 + z * z)


class LogisticTail:
    code, n_par = 2, 1

    def __init__(self, s):
        self.par = (s, mpf(1))

    def H(self, z):
        y = z / self.par[0]
        return 2 * (y + log1p(exp(-2 * y)) - log(2)) if y > 30 else 2 * log(cosh(y))

    def dH(self, z):
        return 2 * tanh(z / self.par[0]) / self.par[0]


# ---------------------------------------------------------------- the two laws


class Mcp:
    code = 0

    def __init__(self, lam, gam, tail):
        self.lam, self.p, self.tail = lam, gam, tail

    def knots(self):
        return [self.p * self.lam]

    def q(self, x):
        K = self.p * self.lam
        if x <= K:
            return self.lam * x - x * x / (2 * self.p)
        return self.p * self.lam ** 2 / 2 + self.tail.H(x - K)

    def dq(self, x):
        K = self.p * self.lam
        if x <= K:
            return self.lam - x / self.p
        return self.tail.dH(x - K)


class Scad:
    code = 1

    def __init__(self, lam, a, tail):
        self.lam, self.p, self.tail = lam, a, tail

    def knots(self):
        return [self.lam, self.p * self.lam]

    def q(self, x):
        K = self.p * self.lam
        if x <= self.lam:
            return self.lam * x
        if x <= K:
            v = x - self.lam
            return self.lam * x - v * v / (2 * (self.p - 1))
        return (self.p + 1) * self.lam ** 2 / 2 + self.tail.H(x - K)

    def dq(self, x):
        K = self.p * self.lam
        if x <= self.lam:
            return self.lam
        if x <= K:
            return self.lam - (x - self.lam) / (self.p - 1)
        return self.tail.dH(x - K)


# ---------------------------------------------------------------- the reference minimizer


def bisect(f, a, b, fa, halvings=190):
    """The root of f in (a, b), to the full working precision; fa = f(a) != 0."""
    for _ in range(halvings):
        m = (a + b) / 2
        if m == a or m == b:
            break
        fm = f(m)
        if fm == 0:
            return m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return (a + b) / 2


def minimize(law, U, t, scan=600):
    """The global minimizer of g over [0, inf), and the gap to the runner-up."""

    def g(x):
        return t * law.q(x) + (x - U) ** 2 / 2

    def dg(x):
        return t * law.dq(x) + x - U

    hi = U + max(law.knots()) + 1
    # candidates: the ends, the knots, and a bisected root of g' in every sign change
    knots = [k for k in law.knots() if 0 < k < hi]
    cand = [mpf(0)] + knots
    # The scan has the knots among its nodes, so that no bracket straddles one: g'
    # jumps there and a sign change across a knot is not a root. A uniform grid alone
    # also misses a root that hugs a knot, which a stationary point of the tail does
    # whenever the point lies just beyond it, so each knot carries a geometric
    # refinement on both sides.
    xs = [hi * mpf(i) / scan for i in range(scan + 1)] + knots + [mpf(0), hi]
    for k in knots:
        for j in range(1, 61):
            xs.append(k + (hi - k) / mpf(2) ** j)
            xs.append(k - k / mpf(2) ** j)
    xs = sorted(set(x for x in xs if 0 <= x <= hi))
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if any(a < k < b for k in knots) or not b > a:
            continue
        # q' at a knot is the value of the piece below it, so a bracket starting there
        # is nudged into the piece above, whose root it is meant to find
        if any(a == k for k in knots):
            a = a + (b - a) / mpf(2) ** 80
        fa, fb = dg(a), dg(b)
        if fa == 0:
            cand.append(a)
        elif fa * fb < 0:
            cand.append(bisect(dg, a, b, fa))
    # one candidate per distinct point: a knot and a stationary point of the next piece
    # can coincide, and the scan can bracket the same root twice
    uniq = []
    for x in sorted(x for x in cand if x >= 0):
        if not uniq or x - uniq[-1] > mpf(2) ** -60 * max(1, x):
            uniq.append(x)
    vals = sorted((g(x), x) for x in uniq)
    # the objective can be flat over a whole piece (the inner curvature vanishes at
    # t = gamma, or t = a - 1, and its slope with it): every point of it minimizes, and
    # the working precision decides the order. Take the smallest minimizer, the rule the
    # implementation follows, and let the gap report the flatness.
    tie = mpf(2) ** -170 * max(1, abs(vals[0][0]))
    best = min(x for gv, x in vals if gv <= vals[0][0] + tie)
    rest = [gv for gv, x in vals if x != best and abs(x - best) > mpf(2) ** -60 * max(1, best)]
    gap = (min(rest) - vals[0][0]) / max(1, abs(vals[0][0])) if rest else mpf(1)
    return best, max(gap, mpf(0))


# ---------------------------------------------------------------- the grid

TAILS = [GaussianTail(mpf(1) / 4), GaussianTail(mpf(4)),
         StudentTail(mpf(1) / 2, mpf(7) / 10), StudentTail(mpf(1), mpf(3)),
         StudentTail(mpf(2), mpf(40)), LogisticTail(mpf(3) / 10), LogisticTail(mpf(2))]

LAMBDAS = [mpf(1) / 5, mpf(1), mpf(5)]
SECONDS = [mpf(1) / 2, mpf(3) / 2, mpf(3), mpf(20)]


def distinct(v):
    """The sorted values, one per distinct one, so that no grid row repeats."""
    out = []
    for x in sorted(v):
        if not out or x > out[-1]:
            out.append(x)
    return out


def steps(law, U):
    """Steps spread over the regimes, including the two curvature changes."""
    c = law.p if law.code == 0 else law.p - 1  # the curvature of the inner piece turns at t = c
    out = [mpf(1) / 100, mpf(1) / 2, mpf(1), mpf(10)]
    out += [c / 2, c, 2 * c]  # convex, flat, concave
    return distinct([s for s in out if s > 0])


def points(law):
    """U over the pieces and the thresholds, where the branch changes."""
    K = law.p * law.lam
    out = []
    for f in (mpf(1) / 100, mpf(1) / 2, mpf(99) / 100, mpf(1), mpf(101) / 100, mpf(2), mpf(10)):
        out.append(f * K)
    out += [law.lam / 2, law.lam, 2 * law.lam, 100 * K, 10000 * K]
    return distinct([u for u in out if u > 0])


def main():
    rows = []
    for tail in TAILS:
        for lam in LAMBDAS:
            for second in SECONDS:
                for law in (Mcp(lam, second, tail), Scad(lam, 1 + second, tail)):
                    for U in points(law):
                        for t in steps(law, U):
                            x, gap = minimize(law, U, t)
                            hi, lo = hex_pair(x)
                            rows.append("%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%.17g" % (
                                law.code, tail.code, float(law.lam).hex(), float(law.p).hex(),
                                float(tail.par[0]).hex(), float(tail.par[1]).hex(),
                                float(U).hex(), float(t).hex(), hi, lo, float(gap)))
    with open(OUT, "w", newline="\n") as f:
        f.write("law,tail,lam,p,t1,t2,U,t,hi,lo,gap\n")
        f.write("\n".join(rows))
        f.write("\n")
    print("%d rows -> %s" % (len(rows), os.path.normpath(OUT)))


if __name__ == "__main__":
    main()
