"""Check the reference manual for markup that Doxygen or MathJax did not render.

Usage (from the repository root, after `doxygen docs/manual/Doxyfile`):

    python -I tools/docs/check_manual.py [build-docs/html]

Two kinds of check:
- the generated HTML pages: raw bold or italic markers, raw Doxygen commands
  (\\f$, \\ref, \\see, \\param) and raw LaTeX left in the text;
- the sources (docs/manual/pages and the documentation comments of
  include/laplace): mathematics in headings, \\text{...} with an underscore
  (MathJax prints the backslash), formulas or code spans inside bold markup.
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
    "raw bold/italic": re.compile(r"\*\*|(?<![\w*])\*(?=\S)[^*\n]{1,80}\*(?![\w*])"),
    "raw formula": re.compile(r"\\f[\$\[\]]|\\(frac|nu|log|mathrm|mu|text)\b"),
    "raw command": re.compile(r"\\(ref|see|param|return|brief)\b"),
}
pages = sorted(p for p in out.glob("*.html") if not p.name.endswith("_source.html"))
for p in pages:
    text = p.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>|<pre.*?</pre>|<code.*?</code>", " ", text, flags=re.S)
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
bold_code = re.compile(r"\*\*[^*\n]*(?:`|\\f\$)[^*\n]*\*\*")
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
        if bold_code.search(line):
            problems.append(f"{rel}:{i}: formula or code inside bold: {line.strip()[:100]!r}")

for line in problems:
    print(line)
print(f"{len(problems)} problem(s) in {len(pages)} pages and {len(sources) + len(headers)} sources")
sys.exit(1 if problems else 0)
