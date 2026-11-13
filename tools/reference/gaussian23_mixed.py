"""Check of the mixed (mu, w) derivatives of a function of z = (y - mu) w^(-k), the route of
detail_location_scale::mixed_power_dk (sigma = w^k: w = v and k = 1/2 for gaussian2,
w = tau and k = -1/2 for gaussian3).

Usage (from the repository root):  python -I tools/reference/gaussian23_mixed.py

For a test function Lambda(z) = exp(sin(z)) + z^3/7 (not polynomial, so every term is
visible), the exact d^a/dmu^a d^b/dw^b Lambda(z(mu, w)) is compared with two recursions:

  current (the one in location_scale.hpp): after the a derivatives in mu,
      R_0 = (-1)^a Lambda^(a)(z), then b steps R <- -(a R + z R') with the factor k/w
      each step, prefactor sigma^(-a);
  corrected: the same start, then b steps R <- -((k a + i) R + k z R') with i the step
      index and the factor 1/w each step (the w-power of the prefactor grows by one per
      step), total sigma^(-a) w^(-b) R_b(z).

The first derivative in w is the same in both. From the second on the current recursion
misses the growth of the power of w (the constant a should be k a + i) and the factor k
of the z-term; the output of this script is the table of differences.
"""
import sys

import sympy as sp

mu, y, w = sp.symbols("mu y w", positive=True)
z_sym = sp.symbols("z", real=True)
LAM = sp.exp(sp.sin(z_sym)) + z_sym**3 / 7


def lam_derivs(order):
    out = [LAM]
    for _ in range(order):
        out.append(sp.diff(out[-1], z_sym))
    return out


def run(k, a_max=3, b_max=3, wval=sp.Rational(13, 10), muval=sp.Rational(2, 5), yval=sp.Rational(14, 10)):
    kk = sp.Rational(k)
    z_expr = (y - mu) * w ** (-kk)
    worst_cur = worst_cor = 0.0
    rows = []
    for a in range(a_max + 1):
        for b in range(b_max + 1):
            true_expr = sp.diff(LAM.subs(z_sym, z_expr), mu, a, w, b) if (a or b) else LAM.subs(z_sym, z_expr)
            true = sp.N(true_expr.subs({y: yval, mu: muval, w: wval}), 30)
            # the z-form after the a derivatives in mu: (-d/dz)^a Lambda
            R0 = (-1) ** a * lam_derivs(a)[a]
            # current recursion (mixed_power_dk): the factor k/w per step, the constant a, z R' with coefficient 1
            Rc = R0
            for _ in range(b):
                Rc = -(a * Rc + z_sym * sp.diff(Rc, z_sym))
            cur = sp.N((wval ** (-kk * a)) * (kk / wval) ** b * Rc.subs(z_sym, (yval - muval) * wval ** (-kk)), 30)
            # corrected recursion
            Rr = R0
            for i in range(b):
                Rr = -((kk * a + i) * Rr + kk * z_sym * sp.diff(Rr, z_sym))
            cor = sp.N((wval ** (-kk * a)) * (1 / wval) ** b * Rr.subs(z_sym, (yval - muval) * wval ** (-kk)), 30)
            e_cur = abs(cur - true) / max(abs(true), 1e-30)
            e_cor = abs(cor - true) / max(abs(true), 1e-30)
            worst_cur = max(worst_cur, float(e_cur))
            worst_cor = max(worst_cor, float(e_cor))
            rows.append((a, b, float(true), float(cur), float(cor)))
    return rows, worst_cur, worst_cor


def main():
    ok = True
    for k, name in ((sp.Rational(1, 2), "gaussian2 (w = v, k = 1/2)"), (sp.Rational(-1, 2), "gaussian3 (w = tau, k = -1/2)")):
        rows, wc, wr = run(k)
        print(f"{name}:")
        for a, b, t, c, r in rows:
            if a + b == 0:
                continue
            print(f"  a={a} b={b}: exact {t: .10g}  current {c: .10g}  corrected {r: .10g}")
        print(f"  worst relative error: current {wc:.3e}, corrected {wr:.3e}")
        ok &= wr < 1e-20
    print("corrected recursion agrees with the exact derivatives" if ok else "CORRECTED RECURSION DISAGREES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
