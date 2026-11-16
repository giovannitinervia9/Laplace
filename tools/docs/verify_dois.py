"""Verify proposed references against Crossref before they enter the manual.

Usage (repository root):

    python -I tools/docs/verify_dois.py FINDINGS.json [FINDINGS.json ...] [--out verified.json]

Each input file is a JSON list of findings, each with a "reference" object:
{"authors": "N. M. Temme", "year": 1975, "title": "...", "venue": "...",
 "doi": "10.1016/0021-9991(75)90082-0" or null, "type": "article | book | software | report"}
(the format of note/manual/brief_citations_scan.md). For each reference:

- a proposed DOI is fetched from Crossref and accepted ("verified") when the
  record's title, first author's surname and year (within one) match;
- otherwise Crossref is searched by title, authors and year, and a hit is
  accepted ("found", or "corrected" when the proposed DOI was wrong) only if
  it matches the same way; peer reviews are skipped, and a book matches only
  a book record (a journal record with a book's title is a review);
- software, references without title or year, and works Crossref does not
  know come out as "no-doi" or "wrong-doi", for a decision by hand (DLMF
  sections are cited by URL; books without a DOI by their standard data).

Writes the findings with "check" and "crossref" fields to --out (default
verified.json) and one line per reference on stdout. Needs network access to
api.crossref.org; sends no personal data (generic User-Agent).
"""
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

UA = {"User-Agent": "laplace-docs-reference-check/1.0"}
BOOKISH = ("book", "monograph", "edited-book", "reference-book", "book-set", "book-series")
cache = {}


def get(url):
    if url in cache:
        return cache[url]
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                cache[url] = json.load(r)
                return cache[url]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                cache[url] = None
                return None
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None


def norm(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def title_ok(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85 or (len(a) > 25 and (a in b or b in a))


def surname(authors):
    # "et al." and "and others" must not leave "al" to match inside another name
    a = re.sub(r"\b(et al\.?|and others)", "", authors or "")
    first = re.split(r",| and |;", a)[0].strip()
    return norm(first.split()[-1]) if first else ""


def record(msg):
    year = None
    for k in ("published-print", "published-online", "issued"):
        dp = (msg.get(k) or {}).get("date-parts") or [[None]]
        if dp[0] and dp[0][0]:
            year = dp[0][0]
            break
    authors = msg.get("author") or msg.get("editor") or []
    return {
        "doi": msg.get("DOI", "").lower(),
        "title": (msg.get("title") or [""])[0],
        "container": (msg.get("container-title") or [""])[0],
        "year": year,
        "authors": ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors),
        "volume": msg.get("volume"), "issue": msg.get("issue"), "pages": msg.get("page"),
        "type": msg.get("type"),
    }


def matches(rec, ref):
    ok_title = title_ok(rec["title"], ref.get("title"))
    sn = surname(ref.get("authors"))
    # a record without authors (some books) is judged on title and year
    ok_author = not sn or not rec["authors"] or sn in norm(rec["authors"])
    y = ref.get("year")
    ok_year = not y or not rec["year"] or abs(int(rec["year"]) - int(y)) <= 1
    return ok_title and ok_author and ok_year


def check(ref):
    doi = (ref.get("doi") or "").strip().lower().removeprefix("https://doi.org/")
    if doi:
        m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()"))
        if m:
            rec = record(m["message"])
            if matches(rec, ref):
                return "verified", rec
    if not ref.get("title") or not ref.get("year") or ref.get("type") == "software":
        return ("wrong-doi" if doi else "no-doi"), None
    q = " ".join(str(x) for x in (ref.get("title"), ref.get("authors"), ref.get("year")) if x)
    s = get("https://api.crossref.org/works?rows=20&query.bibliographic=" + urllib.parse.quote(q))
    for item in ((s or {}).get("message") or {}).get("items", []):
        if item.get("type") in ("peer-review", "component", "posted-content"):
            continue
        rec = record(item)
        if ref.get("type") == "book" and rec["type"] not in BOOKISH:
            continue
        if matches(rec, ref) and rec["year"]:
            return ("corrected" if doi else "found"), rec
    return ("wrong-doi" if doi else "no-doi"), None


def main(argv):
    out = "verified.json"
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    findings = []
    for f in argv:
        findings += json.loads(Path(f).read_text(encoding="utf-8"))
    for fi in findings:
        ref = fi.get("reference") or {}
        fi["check"], fi["crossref"] = check(ref)
        doi = (fi["crossref"] or {}).get("doi", ref.get("doi"))
        print(f"{fi['check']:10} {str(fi.get('page', '')).split('/')[-1]:26} "
              f"{str(ref.get('authors', ''))[:30]:30} {ref.get('year')} {doi}", flush=True)
        time.sleep(0.2)
    Path(out).write_text(json.dumps(findings, indent=1, ensure_ascii=False), encoding="utf-8")
    print(dict(Counter(f["check"] for f in findings)))


if __name__ == "__main__":
    main(sys.argv[1:])
