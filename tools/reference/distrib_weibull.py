"""Reference values for weibull_distrib (mean m, shape sigma): the log-density, log F and log(1 - F)
with their derivatives in (m, sigma) to order 6, the mixed derivatives in y, and the expected
information.

Usage (from the repository root):  python -I tools/reference/distrib_weibull.py [logpdf|log_cdf|log_ccdf|dy|info]

Files in tests/data/reference/, one row per value, all real columns in hexadecimal:
    weibull_logpdf.csv, weibull_log_cdf.csv, weibull_log_ccdf.csv:  m,sigma,y,K,b,hi,lo
        the derivative of order K, b of its indices in sigma and K - b in m (K = 0 is the value);
    weibull_dy.csv:  m,sigma,y,J,K,b,hi,lo
        d^(J+K) log f / dy^J dm^(K-b) dsigma^b, J >= 1, J + K <= 6;
    weibull_info.csv:  m,sigma,mm_hi,mm_lo,ms_hi,ms_lo,ss_hi,ss_lo
        the expected information I_mm, I_m,sigma, I_sigma,sigma.
hi is the double nearest to the mpmath value and lo the difference to the exact value.

Method. Everything comes from the DEFINITION of the family, built from mpmath's own gamma and exp,
and shares no formula with the library (no Bell polynomial, no polygamma, no closed information):
    k = sigma,  b = m / Gamma(1 + 1/sigma),  f(y) = (k/b) (y/b)^(k-1) exp(-(y/b)^k),
    F(y) = -expm1(-(y/b)^k),  and  log f,  log F = log(-expm1(-T)),  log(1 - F) = log(exp(-T)),
with T = (y/b)^k the cumulative hazard. The derivatives are mpmath.diff (finite differences with
its own step control) in the variables m (1 + s), sigma (1 + t), y (1 + u), that is in the
RELATIVE steps s, t, u, so that the step suits a parameter of any size, each divided back by
m^a sigma^b y^j. Each value is computed at 190 and 240 bits and a row is kept when the two agree
to 2^-80 relative.

The expected information is the quadrature of the squared score, not the closed form: with
T = (y/b)^k a standard exponential variable, E[s_i s_j] = integral of s_i s_j e^(-T) dT over
(0, infinity), tanh-sinh at 40 digits, and the scores s_m, s_sigma are mpmath.diff of the log
density at each node.

The grid. sigma in SIGMA (1e-3 to 1e4 by decades with 0.5, 1, 1.086 and 2), m in M_VALUES, and y
chosen through the cumulative hazard: y = b T^(1/sigma) for T in T_VALUES, which covers the deep
left tail where T underflows a double (T = 1e-400, 1e-300), the body, the point T = 1 (the
scale b), and the right tail where 1 - F underflows (T = 700, 800), plus the point y = m and, for
sigma > 1, the mode (not for the mixed y derivatives, whose components have roots at those points, where a
scaled error is meaningless; they use T in DY_T instead). A point is dropped when y is not a double in [1e-300, 1e300]: at small sigma the
scale b = m / Gamma(1 + 1/sigma) is far below the smallest double (sigma = 1e-3 has log b = log m -
5912), so only the far right tail is reachable there.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
AGREE = mpmath.mpf(2) ** -80
P_HI, P_LO = 240, 190  # bits of the two computations

SIGMA = [1e-3, 1e-2, 0.1, 0.5, 1.0, 1.086, 2.0, 10.0, 100.0, 1e3, 1e4]
M_VALUES = [1e-3, 2.5e3]
T_VALUES = ["1e-400", "1e-300", "1e-30", "1e-8", "1e-3", "0.05", "0.3", "0.9", "1", "1.1", "3", "15", "80", "700", "800"]
DY_M = [1.0]
DY_T = ["1e-8", "0.05", "0.25", "0.7", "1.4", "3", "20", "150"]


def mp_scale(m, s):
    """log b = log m - lgamma(1 + 1/sigma), built from gamma."""
    return mpmath.log(m / mpmath.gamma(1 + 1 / s))


def hazard(m, s, y):
    b = m / mpmath.gamma(1 + 1 / s)
    return (y / b) ** s


def q_logpdf(m, s, y):
    b = m / mpmath.gamma(1 + 1 / s)
    return mpmath.log((s / b) * (y / b) ** (s - 1) * mpmath.exp(-((y / b) ** s)))


def q_log_cdf(m, s, y):
    return mpmath.log(-mpmath.expm1(-hazard(m, s, y)))


def q_log_ccdf(m, s, y):
    return mpmath.log(mpmath.exp(-hazard(m, s, y)))


QUANTITY = {"logpdf": q_logpdf, "log_cdf": q_log_cdf, "log_ccdf": q_log_ccdf}


def derivative(q, m, s, y, j, a, b, prec):
    """d^(j + a + b) q / dy^j dm^a dsigma^b at 'prec' bits, by relative steps."""
    with mpmath.workprec(prec):
        m0, s0, y0 = mpmath.mpf(m), mpmath.mpf(s), mpmath.mpf(y)
        if j + a + b == 0:
            return q(m0, s0, y0)
        g = lambda u, t, v: q(m0 * (1 + u), s0 * (1 + t), y0 * (1 + v))
        d = mpmath.diff(g, (0, 0, 0), (a, b, j))
        return d / (m0 ** a * s0 ** b * y0 ** j)


def keep(c, a):
    if c == 0 or abs(a - c) > AGREE * abs(c):
        return None
    hi = float(c)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    return hi, float(c - mpmath.mpf(hi))


def points(ms, ts, with_mode=True, with_m=True):
    """The (m, sigma, y) of the grid; y is a double."""
    out = []
    with mpmath.workprec(300):
        for m in ms:
            for s in SIGMA:
                lb = mp_scale(mpmath.mpf(m), mpmath.mpf(s))
                cands = [mpmath.exp(lb + mpmath.log(mpmath.mpf(t)) / s) for t in ts]
                if with_m:
                    cands.append(mpmath.mpf(m))
                if with_mode and s > 1:
                    cands.append(mpmath.exp(lb + mpmath.log((s - 1) / s) / s))
                for y in cands:
                    if not (mpmath.mpf(1e-300) <= y <= mpmath.mpf(1e300)):
                        continue
                    yd = float(y)
                    out.append((m, s, yd))
    return sorted(set(out))


def task_density(args):
    name, (m, s, y) = args
    q = QUANTITY[name]
    rows, dropped = [], 0
    for K in range(0, 7):
        for b in range(K + 1):
            r = keep(derivative(q, m, s, y, 0, K - b, b, P_HI), derivative(q, m, s, y, 0, K - b, b, P_LO))
            if r is None:
                dropped += 1
                continue
            rows.append(f"{m.hex()},{s.hex()},{y.hex()},{K},{b},{r[0].hex()},{r[1].hex()}\n")
    return rows, dropped


def task_dy(args):
    m, s, y = args
    rows, dropped = [], 0
    for J in range(1, 7):
        for K in range(0, 7 - J):
            for b in range(K + 1):
                r = keep(derivative(q_logpdf, m, s, y, J, K - b, b, P_HI), derivative(q_logpdf, m, s, y, J, K - b, b, P_LO))
                if r is None:
                    dropped += 1
                    continue
                rows.append(f"{m.hex()},{s.hex()},{y.hex()},{J},{K},{b},{r[0].hex()},{r[1].hex()}\n")
    return rows, dropped


def write(fname, header, results):
    n = dropped = 0
    with open(REF / fname, "w", newline="\n") as f:
        f.write(header + "\n")
        for rows, d in results:
            f.writelines(rows)
            n += len(rows)
            dropped += d
    print(f"{fname}: {n} rows ({dropped} dropped)")


def info_row(args):
    m, s = args
    with mpmath.workdps(40):
        m0, s0 = mpmath.mpf(m), mpmath.mpf(s)
        lb = mp_scale(m0, s0)
        cache = {}

        def scores(T):
            if T in cache:
                return cache[T]
            y = mpmath.exp(lb + mpmath.log(T) / s0)
            sm = mpmath.diff(lambda u: q_logpdf(m0 * (1 + u), s0, y), 0) / m0
            ss = mpmath.diff(lambda t: q_logpdf(m0, s0 * (1 + t), y), 0) / s0
            cache[T] = (sm, ss)
            return cache[T]

        def integral(i, j):
            f = lambda T: scores(T)[i] * scores(T)[j] * mpmath.exp(-T)
            return mpmath.quad(f, [0, 0.5, 2, 8, 30, 100, mpmath.inf])

        vals = [integral(0, 0), integral(0, 1), integral(1, 1)]
    cols = []
    for v in vals:
        hi = float(v)
        cols += [hi.hex(), float(v - mpmath.mpf(hi)).hex()]
    return f"{m.hex()},{s.hex()}," + ",".join(cols) + "\n", 0


def main():
    what = sys.argv[1:] or ["logpdf", "log_cdf", "log_ccdf", "dy", "info"]
    with Pool(4) as pool:
        for name in what:
            if name in QUANTITY:
                pts = points(M_VALUES, T_VALUES)
                res = pool.map(task_density, [(name, p) for p in pts], chunksize=2)
                write(f"weibull_{name}.csv", "m,sigma,y,K,b,hi,lo", res)
            elif name == "dy":
                pts = points(DY_M, DY_T, with_mode=False, with_m=False)
                res = pool.map(task_dy, pts, chunksize=1)
                write("weibull_dy.csv", "m,sigma,y,J,K,b,hi,lo", res)
            elif name == "info":
                res = pool.map(info_row, [(m, s) for m in M_VALUES for s in SIGMA], chunksize=1)
                write("weibull_info.csv", "m,sigma,mm_hi,mm_lo,ms_hi,ms_lo,ss_hi,ss_lo", [([r[0]], 0) for r in res])


if __name__ == "__main__":
    sys.exit(main())
