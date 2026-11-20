"""Reference values for the expected information of mcp_distrib and scad_distrib over
the three properizing tails, at order zero and at 200 bits.

  python -I distrib_mcp_scad_info.py [path]

writes tests/data/reference/distrib_mcp_scad_info.csv with one row per law, tail,
parameter setting and pair of parameters:

  law,tail,i,j,hi,lo,lam,p,t1,t2

i and j being the parameter indices (0 the location mu, 1 the rate lambda, 2 gamma for
mcp or a for scad, 3 and 4 the tail's own), hi and lo the double-double value of I_ij
and lam, p, t1, t2 the parameters as C99 hex floats; t2 is zero for the one-parameter
tails. The reference comes from the definition rather than from the closed forms the
library evaluates: I_mu_mu is the quadrature of E[q'(u)^2], the score form, which is the
one that carries the point mass of the kink at the location, I_mu_theta is zero by
symmetry, and the smooth block is the quadrature of E[q_a q_b] minus the product of the
derivatives of log Z, those taken by mpmath from the closed log Z. The script also checks
the closed assembly of the library (the hessian form, E[q_ab] + d2 log Z with the moments
of the inner piece and the tail's three integrals) against that reference, and the five
z-integrals of each tail against the quadrature of their own definitions.
"""
import io
import os
import sys

from mpmath import (mp, mpf, quad, sqrt, pi, exp, expm1, log, log1p, cosh, sech, tanh, gamma,
                    beta, digamma, inf, diff, fabs, erfi, nstr)

mp.prec = 200

# ---------------------------------------------------------------- the tails
# ----------------------------------------------------------------- the tails
# each tail: psi(k, z), kappa(b), varsigma, exponent(j), param(j), and the four
# closed z-integrals S_0, S_Psi, P_0, S_2 (quadrature versions for the check)
class Gaussian:
    name = "gaussian"
    np_ = 1
    def __init__(self, V): self.p = [mpf(V)]
    def psi(self, k, z):
        return z * z / 2 if k == 0 else (z if k == 1 else (1 if k == 2 else 0))
    def kappa(self, b): return mpf(1) if b == 0 else mpf(0)
    def varsigma(self): return sqrt(self.p[0])
    def exponent(self, j): return mpf(1) / 2
    def S0(self): return sqrt(pi / 2)
    def SPsi(self): return sqrt(pi / 2) / 2
    def P0(self): return sqrt(pi / 2)
    def P2(self): return mpf(1)
    def S2(self): return sqrt(pi / 2)

class Logistic:
    name = "logistic"
    np_ = 1
    def __init__(self, s): self.p = [mpf(s)]
    def psi(self, k, z):
        if k == 0: return 2 * log(cosh(z))
        if k == 1: return 2 * tanh(z)
        if k == 2: return 2 * sech(z) ** 2
        raise AssertionError
    def kappa(self, b): return mpf(1) if b == 0 else mpf(0)
    def varsigma(self): return self.p[0]
    def exponent(self, j): return mpf(1)
    def S0(self): return mpf(1)
    def SPsi(self): return 2 * (1 - log(2))
    def P0(self): return mpf(4) / 3
    def P2(self): return (4 * log(2) - 1) / 3
    def S2(self): return pi ** 2 / 9 - mpf(2) / 3

class Student:
    name = "student"
    np_ = 2
    def __init__(self, s, nu): self.p = [mpf(s), mpf(nu)]
    def psi(self, k, z):
        if k == 0: return log1p(z * z)
        if k == 1: return 2 * z / (1 + z * z)
        if k == 2: return 2 * (1 - z * z) / (1 + z * z) ** 2
        raise AssertionError
    def k_(self): return (self.p[1] + 1) / 2
    def kappa(self, b): return self.k_() if b == 0 else (mpf(1) / 2 if b == 1 else mpf(0))
    def varsigma(self): return self.p[0] * sqrt(self.p[1])
    def exponent(self, j): return mpf(1) if j == 0 else mpf(1) / 2
    def S0(self): return sqrt(pi) / 2 * gamma(self.k_() - mpf(1) / 2) / gamma(self.k_())
    def SPsi(self):
        k = self.k_()
        return self.S0() * (digamma(k) - digamma(k - mpf(1) / 2))
    def P0(self):
        k = self.k_()
        return beta(mpf(1) / 2, k + mpf(3) / 2) - beta(mpf(3) / 2, k + mpf(1) / 2)
    def P2(self):
        k = self.k_()
        return (k - 1) / (k * (k + 1))
    def S2(self):
        k = self.k_()
        return beta(mpf(3) / 2, k + mpf(1) / 2) - beta(mpf(5) / 2, k - mpf(1) / 2)

def tail_H(t, r, x):
    """d^r_x H(x), H = kappa Psi(x/varsigma)"""
    vs, k = t.varsigma(), t.kappa(0)
    return k * t.psi(r, x / vs) / vs ** r

def check_tail_integrals(t):
    vs, k = t.varsigma(), t.kappa(0)
    f = lambda z: exp(-k * t.psi(0, z))
    got = [t.S0(), t.SPsi(), t.P0(), t.S2()]
    got.append(t.P2())
    want = [quad(lambda z: f(z), [0, inf]),
            quad(lambda z: t.psi(0, z) * f(z), [0, inf]),
            quad(lambda z: t.psi(2, z) * f(z), [0, inf]),
            quad(lambda z: z * z * t.psi(2, z) * f(z), [0, inf]),
            quad(lambda z: z * t.psi(2, z) * f(z), [0, inf])]
    return max(fabs(a - b) / fabs(b) for a, b in zip(got, want))

# ------------------------------------------------------- the tail's own blocks
def tail_T(t): return t.varsigma() * t.S0()
def tail_J(t): return t.kappa(0) / t.varsigma() * t.P0()

def omega(t, j): return t.exponent(j) / t.p[j]

def tail_B(t, j):
    """int d_theta_j H' e^-H = kappa_j P_1 - kappa omega_j (P_2 + P_1), P_1 = 1/kappa"""
    kt = t.kappa(1) if (isinstance(t, Student) and j == 1) else mpf(0)
    k = t.kappa(0)
    return kt / k - k * omega(t, j) * (t.P2() + 1 / k)

def tail_C(t, i, j):
    """int d2_theta_i theta_j H e^-H"""
    vs, k = t.varsigma(), t.kappa(0)
    S1 = t.S0() / k
    ki = t.kappa(1) if (isinstance(t, Student) and i == 1) else mpf(0)
    kj = t.kappa(1) if (isinstance(t, Student) and j == 1) else mpf(0)
    wi, wj = omega(t, i), omega(t, j)
    v = -(ki * wj + kj * wi) * S1 + k * wi * wj * (S1 + t.S2())
    if i == j: v += k * wi * S1 / t.p[i]
    return vs * v

def _sub(t, j, p):
    q = [t.p[w] for w in range(t.np_)]
    q[j] = p
    return type(t)(*q)

def check_tail_BC(t):
    """B against int d_t H H' e^-H (by parts), C against one mpmath derivative of the
    analytic d_t H, both integrated by quadrature"""
    f = lambda x: exp(-tail_H(t, 0, x))
    def dH(tt, j, x):  # analytic d_theta_j H
        z = x / tt.varsigma()
        kt = tt.kappa(1) if (isinstance(tt, Student) and j == 1) else mpf(0)
        return kt * tt.psi(0, z) - tt.kappa(0) * omega(tt, j) * z * tt.psi(1, z)
    worst = mpf(0)
    for j in range(t.np_):
        want = quad(lambda x: dH(t, j, x) * tail_H(t, 1, x) * f(x), [0, inf])
        worst = max(worst, fabs(tail_B(t, j) - want) / max(fabs(want), mpf(1)))
    for i in range(t.np_):
        for j in range(i, t.np_):
            want = quad(lambda x: diff(lambda p: dH(_sub(t, j, p), i, x), t.p[j]) * f(x),
                        [0, inf])
            worst = max(worst, fabs(tail_C(t, i, j) - want) / max(fabs(want), mpf(1)))
    return worst

# ----------------------------------------------------------------- the laws
def dawson(x):
    x = mpf(x)
    return sqrt(pi) / 2 * exp(-x * x) * erfi(x)


def N_moments(lam, g):
    """N_k = int_0^(g lam) t^k exp(-lam t + t^2/(2g)) dt, k = 0, 1, 2"""
    lam, g = mpf(lam), mpf(g)
    K, E = g * lam, exp(-g * lam * lam / 2)
    G = sqrt(2 * g) * dawson(lam * sqrt(g / 2))
    return [G, K * G - g * (1 - E), (K * K - g) * G + g * K * (2 * E - 1)]


def m_moments(lam):
    """m_k = int_0^lam u^k exp(-lam u) du, k = 0, 1, 2"""
    lam = mpf(lam)
    e = exp(-lam * lam)
    return [-expm1(-lam * lam) / lam,
            (1 - e * (1 + lam * lam)) / lam ** 2,
            (2 - e * (2 + 2 * lam * lam + lam ** 4)) / lam ** 3]


def tail_d1_H(t, i, x):
    """d_theta_i H, analytic"""
    z = x / t.varsigma()
    kt = t.kappa(1) if (isinstance(t, Student) and i == 1) else mpf(0)
    return kt * t.psi(0, z) - t.kappa(0) * omega(t, i) * z * t.psi(1, z)


class Mcp:
    name = "mcp"

    def __init__(self, lam, g, tail):
        self.lam, self.g, self.t = mpf(lam), mpf(g), tail

    def knots(self):
        return [self.g * self.lam]

    def q(self, u):
        lam, g = self.lam, self.g
        if u <= g * lam:
            return lam * u - u * u / (2 * g)
        return g * lam * lam / 2 + tail_H(self.t, 0, u - g * lam)

    def q_prime(self, u):
        lam, g = self.lam, self.g
        if u <= g * lam:
            return lam - u / g
        return tail_H(self.t, 1, u - g * lam)

    def q_smooth_d1(self, j, u):
        # j = 0 is lambda, 1 is gamma, 2 + i is the tail parameter i
        lam, g, t = self.lam, self.g, self.t
        if u <= g * lam:
            if j == 0:
                return u
            if j == 1:
                return u * u / (2 * g * g)
            return mpf(0)
        x = u - g * lam
        hp = tail_H(t, 1, x)
        if j == 0:
            return g * lam - g * hp
        if j == 1:
            return lam * lam / 2 - lam * hp
        return tail_d1_H(t, j - 2, x)

    def z_over_two(self):
        return N_moments(self.lam, self.g)[0] + exp(-self.g * self.lam ** 2 / 2) * tail_T(self.t)

    def int_q_dd(self):
        # int_0^inf q'' e^-q du, the a.e. second derivative
        N = N_moments(self.lam, self.g)
        return -N[0] / self.g + exp(-self.g * self.lam ** 2 / 2) * tail_J(self.t)

    def int_q_d2(self, i, j):
        # int_0^inf q_ij e^-q du over the smooth indices
        lam, g, t = self.lam, self.g, self.t
        N = N_moments(lam, g)
        E = exp(-g * lam * lam / 2)
        T, J = tail_T(t), tail_J(t)
        if i == 0 and j == 0:
            return E * (g * T + g * g * J)
        if i == 0 and j == 1:
            return E * (lam * T - 1 + g * lam * J)
        if i == 1 and j == 1:
            return -N[2] / g ** 3 + E * lam * lam * J
        if i == 0 and j >= 2:
            return -g * E * tail_B(t, j - 2)
        if i == 1 and j >= 2:
            return -lam * E * tail_B(t, j - 2)
        return E * tail_C(t, i - 2, j - 2)


class Scad:
    name = "scad"

    def __init__(self, lam, a, tail):
        self.lam, self.a, self.t = mpf(lam), mpf(a), tail

    def c(self):
        return self.a - 1

    def knots(self):
        return [self.lam, self.a * self.lam]

    def q(self, u):
        lam, a, c = self.lam, self.a, self.c()
        if u <= lam:
            return lam * u
        if u <= a * lam:
            return lam * u - (u - lam) ** 2 / (2 * c)
        return (a + 1) * lam * lam / 2 + tail_H(self.t, 0, u - a * lam)

    def q_prime(self, u):
        lam, a, c = self.lam, self.a, self.c()
        if u <= lam:
            return lam
        if u <= a * lam:
            return lam - (u - lam) / c
        return tail_H(self.t, 1, u - a * lam)

    def q_smooth_d1(self, j, u):
        lam, a, c, t = self.lam, self.a, self.c(), self.t
        if u <= lam:
            return u if j == 0 else mpf(0)
        if u <= a * lam:
            if j == 0:
                return u + (u - lam) / c
            if j == 1:
                return (u - lam) ** 2 / (2 * c * c)
            return mpf(0)
        x = u - a * lam
        hp = tail_H(t, 1, x)
        if j == 0:
            return (a + 1) * lam - a * hp
        if j == 1:
            return lam * lam / 2 - lam * hp
        return tail_d1_H(t, j - 2, x)

    def z_over_two(self):
        lam, a, c = self.lam, self.a, self.c()
        return (m_moments(lam)[0] + exp(-lam * lam) * N_moments(lam, c)[0] +
                exp(-(a + 1) * lam * lam / 2) * tail_T(self.t))

    def int_q_dd(self):
        lam, a, c = self.lam, self.a, self.c()
        return (-exp(-lam * lam) * N_moments(lam, c)[0] / c +
                exp(-(a + 1) * lam * lam / 2) * tail_J(self.t))

    def int_q_d2(self, i, j):
        lam, a, c, t = self.lam, self.a, self.c(), self.t
        N = N_moments(lam, c)
        e1, E = exp(-lam * lam), exp(-(a + 1) * lam * lam / 2)
        T, J = tail_T(t), tail_J(t)
        if i == 0 and j == 0:
            return -e1 * N[0] / c + E * ((a + 1) * T + a * a * J)
        if i == 0 and j == 1:
            return -e1 * N[1] / c ** 2 + E * (lam * T - 1 + a * lam * J)
        if i == 1 and j == 1:
            return -e1 * N[2] / c ** 3 + E * lam * lam * J
        if i == 0 and j >= 2:
            return -a * E * tail_B(t, j - 2)
        if i == 1 and j >= 2:
            return -lam * E * tail_B(t, j - 2)
        return E * tail_C(t, i - 2, j - 2)


def smooth_params(law):
    second = law.g if law.name == "mcp" else law.a
    return [law.lam, second] + list(law.t.p)


def rebuild(law, vals):
    t = type(law.t)(*vals[2:])
    return type(law)(vals[0], vals[1], t)


def log_z(law):
    return log(2 * law.z_over_two())


def log_z_d(law, i, j=None):
    v = smooth_params(law)
    n = len(v)

    def one(p, w_i):
        return log_z(rebuild(law, [p if w == w_i else v[w] for w in range(n)]))

    if j is None:
        return diff(lambda p: one(p, i), v[i])
    if i == j:
        return diff(lambda p: one(p, i), v[i], 2)

    def two(p, r):
        return log_z(rebuild(law, [p if w == i else (r if w == j else v[w]) for w in range(n)]))

    return diff(two, (v[i], v[j]), (1, 1))


def closed_info(law):
    """the closed assembly, keyed by the law's parameter indices with 0 for mu"""
    n = 2 + law.t.np_
    Z2 = law.z_over_two()
    out = {(0, 0): (law.lam + law.int_q_dd()) / Z2}
    for a in range(n):
        out[(0, a + 1)] = mpf(0)
    for a in range(n):
        for b in range(a, n):
            out[(a + 1, b + 1)] = law.int_q_d2(a, b) / Z2 + log_z_d(law, a, b)
    return out


def reference_info(law):
    """the score form by quadrature of the definition"""
    n = 2 + law.t.np_
    Z2 = law.z_over_two()
    far = law.knots()[-1] + 60 * (law.t.varsigma() + 1 / law.lam)
    panels = [mpf(0)] + law.knots() + [far, inf]

    def E(f):
        return quad(lambda u: f(u) * exp(-law.q(u)), panels) / Z2

    zd = [log_z_d(law, i) for i in range(n)]
    out = {(0, 0): E(lambda u: law.q_prime(u) ** 2)}
    for a in range(n):
        out[(0, a + 1)] = mpf(0)
    for a in range(n):
        for b in range(a, n):
            out[(a + 1, b + 1)] = (E(lambda u: law.q_smooth_d1(a, u) * law.q_smooth_d1(b, u)) -
                                   zd[a] * zd[b])
    return out




# ------------------------------------------------------------- the reference
# the parameter settings: w = p lambda^2 across the two branches of the moments of the
# inner piece, and tails from a narrow to a heavy one
TAIL_SETS = {
    "gaussian": [("0.3",), ("1.0",), ("7.5",)],
    "logistic": [("0.4",), ("1.3",), ("6.0",)],
    "student": [("0.9", "0.7"), ("0.9", "3.5"), ("2.0", "17.0"), ("0.5", "250.0")],
}
LAW_SETS = {
    "mcp": [("0.25", "0.4"), ("0.7", "0.9"), ("1.3", "3.0"), ("2.0", "5.0"), ("4.0", "9.0")],
    "scad": [("0.2", "1.4"), ("0.6", "2.0"), ("1.1", "3.7"), ("1.8", "6.0"), ("3.5", "8.0")],
}
TAIL_TYPES = {"gaussian": Gaussian, "logistic": Logistic, "student": Student}


def hexpair(v):
    hi = float(v)
    lo = float(v - mpf(hi))
    return "%s,%s" % (hi.hex(), lo.hex())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "data", "reference", "distrib_mcp_scad_info.csv")
    rows = ["law,tail,i,j,hi,lo,lam,p,t1,t2"]
    worst_closed, worst_tail, where = mpf(0), mpf(0), None
    for tname, tsets in sorted(TAIL_SETS.items()):
        for tp in tsets:
            t = TAIL_TYPES[tname](*tp)
            worst_tail = max(worst_tail, check_tail_integrals(t), check_tail_BC(t))
    for lname, lsets in sorted(LAW_SETS.items()):
        for p1, p2 in lsets:
            for tname, tsets in sorted(TAIL_SETS.items()):
                for tp in tsets:
                    t = TAIL_TYPES[tname](*tp)
                    law = Mcp(p1, p2, t) if lname == "mcp" else Scad(p1, p2, t)
                    ref, closed = reference_info(law), closed_info(law)
                    scale = max(fabs(v) for v in ref.values())
                    for k in sorted(ref):
                        e = fabs(ref[k] - closed[k]) / scale
                        if e > worst_closed:
                            worst_closed, where = e, (lname, tname, p1, p2, k)
                    pr = [float(law.lam), float(law.g if lname == "mcp" else law.a)]
                    pr += [float(x) for x in t.p] + [0.0] * (2 - t.np_)
                    tail_hex = ",".join(x.hex() for x in pr)
                    for (i, j) in sorted(ref):
                        rows.append("%s,%s,%d,%d,%s,%s" %
                                    (lname, tname, i, j, hexpair(ref[(i, j)]), tail_hex))
                    print("%-5s %-9s %-5s %-5s %s" % (lname, tname, p1, p2,
                                                      " ".join(nstr(x, 4) for x in t.p)))
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")
    print("wrote %s (%d rows)" % (out, len(rows) - 1))
    print("tail integrals against quadrature: %.2e" % float(worst_tail))
    print("closed assembly against the reference: %.2e at %s" % (float(worst_closed), where))


if __name__ == "__main__":
    main()
