"""Check the reference manual for markup that Doxygen or MathJax did not render.

Usage (from the repository root, after `doxygen docs/manual/Doxyfile`):

    python -I tools/docs/check_manual.py [build-docs/html]

Two kinds of check:
- the generated HTML pages: raw bold or italic markers, raw Doxygen commands
  (\\f$, \\ref, \\see, \\param) and raw LaTeX left in the text;
- the sources (docs/manual/pages and the documentation comments of
  include/laplace): mathematics in headings, \\text{...} with an underscore
  (MathJax prints the backslash), formulas or code spans inside bold markup;
- the sidebar hierarchy: docs/manual/pages/groups.dox alone gives each group
  its parent, in the order of the sidebar. A \\defgroup elsewhere that carries
  its own \\ingroup jumps ahead of that order (or gets a second parent), and a
  group defined elsewhere that groups.dox does not attach floats to the top;
- one page per distribution: the struct X_distrib belongs to the group
  distrib_X and to no shared group, so that the sidebar under Distributions
  lists distributions only (tools/docs/trim_navtree.py drops the other types).
Exits with status 1 when a problem is found.
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build-docs" / "html"

problems = []

# ---------------------------------------------------------------- generated HTML
raw = {
    "raw bold/italic": re.compile(r"\*\*|(?<![\w*])\*(?=\S)[^*\n]{0,79}\S\*(?![\w*])"),  # closing * after a non-space, as in Markdown (not "double *out, double *)")
    "raw formula": re.compile(r"\\f[\$\[\]]|\\(frac|nu|log|mathrm|mu|text)\b"),
    "raw command": re.compile(r"\\(ref|see|param|return|brief)\b"),
}
pages = sorted(p for p in out.glob("*.html") if not p.name.endswith("_source.html"))
for p in pages:
    text = p.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>|<pre.*?</pre>|<code.*?</code>", " ", text, flags=re.S)
    # code fragments (\code ... \endcode) and function signatures hold C++ such as double**
    text = re.sub(r'<div class="fragment">.*?</div><!-- fragment -->', " ", text, flags=re.S)
    text = re.sub(r'<table class="(?:memname|mlabels)">.*?</table>', " ", text, flags=re.S)
    text = re.sub(r'<td class="(?:memItemRight|memItemLeft|memTemplParams)"[^>]*>.*?</td>', " ", text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", " ", text, flags=re.S)  # MathJax formulas
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    for name, rx in raw.items():
        for m in rx.finditer(text):
            s = max(0, m.start() - 50)
            problems.append(f"{p.name}: {name}: ...{text[s:m.end() + 50]!r}")

# ---------------------------------------------------------------- sources
heading = re.compile(r"^\s*(?:\*\s*|///\s*)?#{1,6}\s+(.*)$")
mathy = re.compile(r"\\f\$|\b(?:x|y|z|t|nu|mu|K_\w+|log K\w*)\b|[_^]")
text_us = re.compile(r"\\text\{[^}]*_[^}]*\}")
doc_line = re.compile(r"^\s*(?:\*|///|/\*\*)")

sources = sorted((ROOT / "docs" / "manual" / "pages").glob("*.md"))
sources += sorted((ROOT / "docs" / "manual" / "pages").glob("*.dox"))
headers = sorted((ROOT / "include" / "laplace").rglob("*.hpp"))
for p in sources + headers:
    is_header = p.suffix == ".hpp"
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if is_header and not doc_line.match(line):
            continue
        rel = p.relative_to(ROOT).as_posix()
        h = heading.match(line)
        if h and mathy.search(h.group(1)):
            problems.append(f"{rel}:{i}: mathematics in a heading: {h.group(1)!r}")
        if text_us.search(line):
            problems.append(f"{rel}:{i}: \\text with an underscore: {line.strip()[:100]!r}")
        # the text inside each **...** pair (odd pieces after splitting on **)
        pieces = line.split("**")
        if len(pieces) > 2 and any("`" in s or "\\f$" in s for s in pieces[1:-1:2]):
            problems.append(f"{rel}:{i}: formula or code inside bold: {line.strip()[:100]!r}")

groups_dox = ROOT / "docs" / "manual" / "pages" / "groups.dox"
top_level = set(re.findall(r"\\defgroup\s+(\w+)", groups_dox.read_text(encoding="utf-8")))
attached = set(re.findall(r"\\addtogroup\s+(\w+)\s*\n\s*\\ingroup", groups_dox.read_text(encoding="utf-8")))
block = re.compile(r"/\*\*.*?\*/", re.S)
for f in sorted((ROOT / "docs" / "manual" / "pages").glob("*.dox")) + headers:
    if f == groups_dox:
        continue
    rel = f.relative_to(ROOT).as_posix()
    for b in block.findall(f.read_text(encoding="utf-8", errors="replace")):
        d = re.search(r"\\defgroup\s+(\w+)", b)
        if not d:
            continue
        if re.search(r"\\ingroup\b", b):
            problems.append(f"{rel}: \\defgroup {d.group(1)} has its own \\ingroup; give its parent in groups.dox only")
        if d.group(1) not in attached | top_level:
            problems.append(f"{rel}: group {d.group(1)} is not attached in groups.dox (\\addtogroup ... \\ingroup)")

distrib_struct = re.compile(r"/\*\*((?:(?!\*/).)*?)\*/\s*(?:template\b[^{;]*?)?struct\s+(\w+)_distrib\b", re.S)
for f in headers:
    for m in distrib_struct.finditer(f.read_text(encoding="utf-8", errors="replace")):
        g = re.search(r"\\ingroup\s+(\w+)", m.group(1))
        if not g or g.group(1) != "distrib_" + m.group(2):
            got = g.group(1) if g else "no group"
            problems.append(f"{f.relative_to(ROOT).as_posix()}: {m.group(2)}_distrib is in {got}; it belongs on its own page, group distrib_{m.group(2)}")

for line in problems:
    print(line)
print(f"{len(problems)} problem(s) in {len(pages)} pages and {len(sources) + len(headers)} sources")
sys.exit(1 if problems else 0)
