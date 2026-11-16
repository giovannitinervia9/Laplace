"""Check a rewrite of manual text against its committed version.

Usage (repository root):  python -I tools/docs/check_rewrite.py FILE [FILE ...] [--moved MOVED.md]

For each file, compares the documentation text of the working copy with the
version in HEAD (for headers: only documentation comment lines; code lines
must be identical):
- numbers in the new text that were not in the old text (possible invention);
- numbers removed from the text that are found neither in docs/numerics.md
  nor in the --moved file (possible loss);
- for headers, any changed code line.
Exits with status 1 when a code line changed or a number was invented.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = re.compile(r"(?<![\w.^_{])(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w])")
DOC = re.compile(r"^\s*(?:\*|///|/\*\*)")


def old_text(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def doc_and_code(text, is_header):
    if not is_header:
        return text, []
    doc, code = [], []
    for line in text.splitlines():
        (doc if DOC.match(line) else code).append(line)
    return "\n".join(doc), [c for c in code if c.strip()]


TIMES_TEN = re.compile(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?")


def numbers(text):
    # ignore numbers inside identifiers, hex floats and anchors; keep "0.43", "1e-15", "767";
    # compared by value, so that "66" and "66.0" are the same number, and
    # "1.48 \times 10^3" (LaTeX in the pages) is the same number as "1.48e3" (numerics.md)
    text = TIMES_TEN.sub(lambda m: f"{m.group(1)}e{m.group(2)}", text)
    out = set()
    for s in NUM.findall(text):
        try:
            out.add(float(s))
        except ValueError:
            pass
    return out


args = sys.argv[1:]
moved = ""
if "--moved" in args:
    i = args.index("--moved")
    for m in args[i + 1:]:
        if Path(m).exists():
            moved += Path(m).read_text(encoding="utf-8")
    args = args[:i]
numerics = (ROOT / "docs" / "numerics.md").read_text(encoding="utf-8")
bad = False
# numbers of the old text of every file given: a number may move between a page and its entries
old_all = set()
for f in args:
    old_all |= numbers(doc_and_code(old_text(f), f.endswith(".hpp"))[0])
known = old_all | numbers(numerics)
for f in args:
    is_header = f.endswith(".hpp")
    new_doc, new_code = doc_and_code(Path(f).read_text(encoding="utf-8"), is_header)
    old_doc, old_code = doc_and_code(old_text(f), is_header)
    if new_code != old_code:
        bad = True
        print(f"{f}: CODE CHANGED ({len(old_code)} -> {len(new_code)} non-doc lines)")
    n_old, n_new = numbers(old_doc), numbers(new_doc)
    invented = sorted(n_new - known)
    removed = n_old - n_new
    numerics_n, moved_n = numbers(numerics), numbers(moved)
    lost = sorted(x for x in removed if x not in numerics_n and x not in moved_n)
    if invented:
        bad = True
    print(f"{f}: words {len(old_doc.split())} -> {len(new_doc.split())}; "
          f"invented {invented[:30]}; removed {len(removed)}, not in numerics or moved: {lost[:30]}")
sys.exit(1 if bad else 0)
