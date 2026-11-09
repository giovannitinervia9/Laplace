"""Reference values of the inverse incomplete gamma and beta functions,
computed with mpmath at 200 bits.

Usage (from the repository root):  python -I tools/reference/inverse.py [name ...]

Format as in incomplete.py: the arguments, then hi, lo (hexadecimal), with
hi + lo the exact inverse as a double-double. The files complement
gamma_p_inv.csv and ibeta_inv.csv written by incomplete.py:

    gamma_q_inv          a, q        x with Q(a, x) = q; (a, x0) drawn as in incomplete.py,
                                     q = Q(a, x0) rounded to a double
    gamma_p_inv_tail     a, p        x with P(a, x) = p, p log-uniform in [2^-1074, 2^-1]
                                     (subnormal p included)
    gamma_q_inv_tail     a, q        x with Q(a, x) = q, q log-uniform in [2^-1074, 2^-1]
    gamma_p_inv_logp     a, log p    x with log P(a, x) = log p, -log p log-uniform in
                                     [2^-10, 2^17] (beyond the double range of p)
    gamma_q_inv_logq     a, log q    x with log Q(a, x) = log q, likewise
    ibetac_inv           a, b, q     x with 1 - I_x(a, b) = q, (a, b, x0) drawn as in incomplete.py
    ibeta_inv_tail       a, b, p     x with I_x(a, b) = p, p log-uniform in [2^-1074, 2^-1]
    ibetac_inv_tail      a, b, q     x with 1 - I_x(a, b) = q, likewise
    ibeta_inv_logp       a, b, log p x with log I_x(a, b) = log p
    ibetac_inv_logq      a, b, log q x with log(1 - I_x(a, b)) = log q
    <beta name>_c        the same arguments, 1 - x on the rows with x > 1/2
                         (written with each beta file; for incomplete.py's
                         ibeta_inv.csv by the name ibeta_inv_c)

The exact inverse is found by Newton's method on the logarithm of the small
side (P or Q, I or 1 - I) as a function of v = log x (gamma) or
v = logit x = log(x / (1 - x)) (beta). Both functions are concave in v (the
densities of log X and of logit X are log-concave, so are their distribution
and survival functions), so that Newton's iterates approach the root
monotonically after the first step; steps are limited to 4 in v so that a
start far from the root costs iterations instead of overflow. Beta arguments
are carried as the pair (x, 1 - x), both with full relative precision, so
that x near 1 keeps its complement.

The robust formulas for P, Q, I and 1 - I are copied from incomplete.py
(series and Legendre's continued fraction for the gamma, the hypergeometric
series on the side below the mean for the beta), written here in terms of the
pair (x, 1 - x).
"""
import math
import random
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261109
N_GAMMA = 4000
N_BETA = 2000


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


# ---------------------------------------------------------------- gamma (copied from incomplete.py)

def _P_series(a, x):
    return mpmath.exp(a * mpmath.log(x) - x - mpmath.loggamma(a + 1)) * \
        mpmath.hyp1f1(1, a + 1, x, maxterms=10**7)


def _Q_upper(a, x):
    tiny = mpmath.mpf(2) ** -1000
    eps = mpmath.mpf(2) ** -190
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10**7):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return mpmath.exp(a * mpmath.log(x) - x - mpmath.loggamma(a)) * h


def P(a, x):
    if x <= a:
        return _P_series(a, x)
    try:
        return 1 - _Q_upper(a, x)
    except Exception:
        return _P_series(a, x)


def Q(a, x):
    if x > a:
        try:
            return _Q_upper(a, x)
        except Exception:
            pass
    return 1 - _P_series(a, x)


# ---------------------------------------------------------------- beta, on the pair (x, y = 1 - x)

def _lbeta(a, b):
    return mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)


def _I_series(a, b, x, y):
    # I_x(a, b) = x^a y^b / (a B(a, b)) 2F1(a + b, 1; a + 1; x), all terms positive
    return mpmath.exp(a * mpmath.log(x) + b * mpmath.log(y) - mpmath.log(a) - _lbeta(a, b)) * \
        mpmath.hyp2f1(a + b, 1, a + 1, x, maxterms=10**7)


def I(a, b, x, y):
    if x <= a / (a + b):
        return _I_series(a, b, x, y)
    return 1 - _I_series(b, a, y, x)


def Ic(a, b, x, y):
    if x <= a / (a + b):
        return 1 - _I_series(a, b, x, y)
    return _I_series(b, a, y, x)


def pair(u):
    """(x, 1 - x) from u = logit x, both with full relative precision."""
    if u < 0:
        e = mpmath.exp(u)
        return e / (1 + e), 1 / (1 + e)
    e = mpmath.exp(-u)
    return 1 / (1 + e), e / (1 + e)


# ---------------------------------------------------------------- Newton on a concave log F(v)

def solve(logF_and_d, target, v0, increasing):
    """v with log F(v) = target; log F concave and monotone in v. logF_and_d(v)
    returns log F(v) and its derivative."""
    v = mpmath.mpf(v0)
    tol = mpmath.mpf(2) ** -160
    prev = None
    for it in range(3000):
        lf, d = logF_and_d(v)
        g = lf - target
        if d == 0 or not mpmath.isfinite(d):
            step = mpmath.mpf(-4 if (g > 0) == increasing else 4)
        else:
            step = -g / d
        if abs(step) > 4:
            step = 4 * mpmath.sign(step)
        v += step
        if abs(step) <= tol * max(1, abs(v)):
            return v
        # noise of the 200-bit evaluation: far below the double-double target
        if prev is not None and abs(step) < mpmath.mpf(2) ** -130 * max(1, abs(v)) and abs(step) >= abs(prev) / 2:
            return v
        prev = step
    raise RuntimeError("no convergence")


def gamma_inverse(a, target, lower, v0=None):
    """x with log P(a, x) = target (lower) or log Q(a, x) = target."""
    A = mpmath.mpf(a)
    T = mpmath.mpf(target)
    lga = mpmath.loggamma(A)

    def logF_and_d(t):
        x = mpmath.exp(t)
        F = P(A, x) if lower else Q(A, x)
        xf = mpmath.exp(A * t - x - lga)  # x times the density
        return mpmath.log(F), (xf / F if lower else -xf / F)

    if v0 is None:
        if lower:
            # P(a, x) <= x^a / Gamma(a + 1): this start is left of the root
            v0 = (T + mpmath.loggamma(A + 1)) / A
            v0 = min(v0, mpmath.log(A + 1))
        else:
            L = -T
            v0 = mpmath.log(A + mpmath.sqrt(2 * A * L) + L)
    t = solve(logF_and_d, T, v0, increasing=lower)
    return mpmath.exp(t)


def beta_inverse(a, b, target, lower, v0=None):
    """x with log I_x(a, b) = target (lower) or log(1 - I_x(a, b)) = target."""
    A, B = mpmath.mpf(a), mpmath.mpf(b)
    T = mpmath.mpf(target)
    lb = _lbeta(A, B)

    def logF_and_d(u):
        x, y = pair(u)
        F = I(A, B, x, y) if lower else Ic(A, B, x, y)
        D = mpmath.exp(A * mpmath.log(x) + B * mpmath.log(y) - lb)  # x y times the density
        return mpmath.log(F), (D / F if lower else -D / F)

    if v0 is None:
        if lower:
            lx = (T + mpmath.log(A) + lb) / A
            v0 = min(lx, mpmath.log(A / (A + B)))
        else:
            ly = (T + mpmath.log(B) + lb) / B
            v0 = -min(ly, mpmath.log(B / (A + B)))
    u = solve(logF_and_d, T, v0, increasing=lower)
    return pair(u)


# ---------------------------------------------------------------- argument sets

def gamma_args(rng, n):
    # as incomplete.gamma_args
    out = []
    for i in range(n):
        a = log_uniform(rng, -10, 14)
        if i % 2 == 0:
            x = a + rng.uniform(-30.0, 30.0) * math.sqrt(a)
            if x <= 0:
                x = log_uniform(rng, -30, 0) * a
        else:
            x = log_uniform(rng, -30, 15)
        out.append((a, x))
    return out


def beta_args(rng, n):
    # as incomplete.beta_args
    out = []
    for i in range(n):
        a, b = log_uniform(rng, -8, 12), log_uniform(rng, -8, 12)
        m = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        k = i % 4
        if k == 0:
            x = rng.random()
        elif k == 1:
            x = m + rng.uniform(-30.0, 30.0) * sd
        elif k == 2:
            x = log_uniform(rng, -60, -1)
        else:
            x = 1.0 - log_uniform(rng, -53, -1)
        if 0.0 < x < 1.0:
            out.append((a, b, x))
    return out


def tail_p(rng):
    return log_uniform(rng, -1074, -2)  # 2^-1074 .. 2^-1


def tail_logp(rng):
    return -log_uniform(rng, -10, 16)  # -2^17 .. -2^-10


def rows_for(name, rng):
    rows = []
    if name == "gamma_q_inv":
        for a, x0 in gamma_args(rng, N_GAMMA):
            A = mpmath.mpf(a)
            q = float(Q(A, mpmath.mpf(x0)))
            if not (0.0 < q < 1.0):
                continue
            x = gamma_inverse(a, mpmath.log(q), False, v0=math.log(x0))
            rows.append(((a, q), x))
    elif name in ("gamma_p_inv_tail", "gamma_q_inv_tail", "gamma_p_inv_logp", "gamma_q_inv_logq"):
        lower = name.startswith("gamma_p")
        logs = name.endswith("logp") or name.endswith("logq")
        for _ in range(N_GAMMA // 2):
            a = log_uniform(rng, -10, 14)
            s = tail_logp(rng) if logs else tail_p(rng)
            target = s if logs else mpmath.log(s)
            x = gamma_inverse(a, target, lower)
            rows.append(((a, s), x))
    elif name == "ibetac_inv":
        for a, b, x0 in beta_args(rng, N_BETA):
            A, B = mpmath.mpf(a), mpmath.mpf(b)
            X = mpmath.mpf(x0)
            q = float(Ic(A, B, X, 1 - X))
            if not (0.0 < q < 1.0):
                continue
            u0 = math.log(x0) - math.log1p(-x0)
            x = beta_inverse(a, b, mpmath.log(q), False, v0=u0)
            rows.append(((a, b, q), x))
    elif name in ("ibeta_inv_tail", "ibetac_inv_tail", "ibeta_inv_logp", "ibetac_inv_logq"):
        lower = name.startswith("ibeta_")
        logs = name.endswith("logp") or name.endswith("logq")
        for _ in range(N_BETA // 2):
            a, b = log_uniform(rng, -8, 12), log_uniform(rng, -8, 12)
            s = tail_logp(rng) if logs else tail_p(rng)
            target = s if logs else mpmath.log(s)
            x = beta_inverse(a, b, target, lower)
            rows.append(((a, b, s), x))
    else:
        raise ValueError(name)
    return rows


ARGNAMES = {"gamma_q_inv": "a,q", "gamma_p_inv_tail": "a,p", "gamma_q_inv_tail": "a,q",
            "gamma_p_inv_logp": "a,logp", "gamma_q_inv_logq": "a,logq", "ibetac_inv": "a,b,q",
            "ibeta_inv_tail": "a,b,p", "ibetac_inv_tail": "a,b,q", "ibeta_inv_logp": "a,b,logp",
            "ibetac_inv_logq": "a,b,logq", "ibeta_inv_c": "a,b,p"}


def write(path, header, rows):
    with path.open("w", newline="\n") as fh:
        fh.write(header + ",hi,lo\n")
        for args, v in rows:
            hi, lo = dd(v)
            fh.write(",".join(float.hex(float(a)) for a in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
    print(f"{path}: {len(rows)} rows", flush=True)


def complement_of_ibeta_inv():
    """ibeta_inv_c.csv from incomplete.py's ibeta_inv.csv: 1 - x* on the rows
    with x* > 1/2, from the double-double x* (exact there: those rows have
    1 - x* >= 2^-53, far above the 2^-106 resolution of the double-double)."""
    rows = []
    with (OUT / "ibeta_inv.csv").open() as fh:
        next(fh)
        for line in fh:
            v = [float.fromhex(t) for t in line.strip().split(",")]
            x = mpmath.mpf(v[-2]) + mpmath.mpf(v[-1])
            if x > 0.5:
                rows.append((tuple(v[:-2]), 1 - x))
    write(OUT / "ibeta_inv_c.csv", "a,b,p", rows)


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name == "ibeta_inv_c":
            complement_of_ibeta_inv()
            continue
        rng = random.Random(f"{SEED}-{name}")
        rows = rows_for(name, rng)
        if name.startswith("ibeta"):
            # the beta inverses carry the pair (x, 1 - x): x goes to <name>.csv,
            # and 1 - x on the rows with x > 1/2 to <name>_c.csv, where it keeps
            # the relative precision that x near 1 has lost
            write(OUT / f"{name}.csv", ARGNAMES[name], [(args, v[0]) for args, v in rows])
            write(OUT / f"{name}_c.csv", ARGNAMES[name], [(args, v[1]) for args, v in rows if v[0] > 0.5])
        else:
            write(OUT / f"{name}.csv", ARGNAMES[name], rows)


if __name__ == "__main__":
    main(sys.argv[1:] or list(ARGNAMES))
