"""Check a citation pass on the manual pages against their committed version.

Usage (repository root):  python -I tools/docs/check_citations.py [PAGE ...]

Without arguments, checks every page of docs/manual/pages changed against HEAD.
For each page:
- every URL in the page must be in a verified list of note/manual/citations/
  (or already in the committed page);
- outside the "## References" sections, the words may change only by inserted
  citations (a year, DLMF, a software name, "and others"); any other insertion,
  deletion or replacement is printed for a look by hand ("check" marks a
  replacement that looks like a citation added to a word, "insert"/"delete"/
  "replace" anything else).
Exits with status 1 when a page has an unverified URL or a non-citation change.
"""
import difflib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = re.compile(r"https?://(?:[^\s()<>\]]|\([^\s()<>]*\))+")
CITE = re.compile(r"\d{4}|DLMF|§|mpmath|Boost|Eigen|fdlibm|SymPy|and others|;")


def urls(text):
    return {u.rstrip(".,") for u in URL.findall(text)}


def strip_refs(text):
    return re.sub(r"## References.*?(?=\n\s*\*/|\n#|\n\s*\*?\s*\\defgroup|\Z)", "", text, flags=re.S)


verified = set()
for f in (ROOT / "note/manual/citations").glob("*.md"):
    verified |= urls(f.read_text(encoding="utf-8"))

pages = sys.argv[1:] or subprocess.run(["git", "diff", "--name-only", "docs/manual/pages"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.split()
bad = 0
for rel in pages:
    rel = Path(rel).resolve().relative_to(ROOT).as_posix()
    new = (ROOT / rel).read_text(encoding="utf-8")
    old = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8").stdout
    unknown = sorted(urls(new) - verified - urls(old))
    a, b = strip_refs(old).split(), strip_refs(new).split()
    issues = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        removed, added = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if op == "insert" and CITE.search(added) and len(added) < 120:
            continue
        kind = "check" if op == "replace" and CITE.search(added) and len(added) < 160 and len(removed) < 80 else op
        issues.append((kind, removed, added))
    if unknown or any(k != "check" for k, _, _ in issues):
        bad += 1
    print(f"== {rel}: {len(issues)} changes to look at, unverified URLs {unknown}")
    for kind, r, ad in issues:
        print(f"   {kind}: -[{r[:160]}] +[{ad[:200]}]")
print("pages with problems:", bad)
sys.exit(1 if bad else 0)
