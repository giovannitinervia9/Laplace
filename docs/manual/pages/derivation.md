# How formulas are derived {#derivation}

Every numerical routine in Laplace is written directly in C++, with explicit
derivatives and one function per derivative order. The library uses no automatic
differentiation and no dual-number or jet type, whether at compile time or at run
time, and no code is generated at build time: all formulas are written directly in
the library headers.

## Symbolic derivation

Where a formula is long, involving many terms across multiple derivative orders,
it is derived symbolically using SymPy (version 1.14.0; Meurer and others 2017),
and the result is transcribed into C++ by hand. The derivation scripts live in
`tools/reference/` and are executed in isolated mode with `python -I`. Each
script also checks the closed forms it produces against SymPy's own derivative
of the defining expression. The module page of a function whose formulas were
derived this way names the script.

## Checking the transcription

The transcription is checked by tests against reference values computed at high
precision (see \ref accuracy "How accuracy is measured"). The reference values
come from a different route: the function is written in SymPy from its
definition, differentiated symbolically, and evaluated with mpmath (Johansson and
others), so the reference does not depend on the transcribed closed form. A sign
error, a miscopied coefficient or an omitted term in the C++ code therefore shows
up as a failed test.

## References

- A. Meurer and others (2017), SymPy: symbolic computing in Python, PeerJ Computer Science 3:e103.
- F. Johansson and others, mpmath: a Python library for arbitrary-precision floating-point arithmetic, version 1.3.0, https://mpmath.org/
