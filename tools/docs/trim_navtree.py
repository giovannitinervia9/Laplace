"""Keep only modules and types in the sidebar of the reference manual.

Usage (repository root, after `doxygen docs/manual/Doxyfile`):

    python -I tools/docs/trim_navtree.py [build-docs/html]

Doxygen lists every member of a group (functions, variables, typedefs, and the
members of each struct) in the navigation tree, which buries the modules under
hundreds of entries. This rewrites each group__*.js so that a group keeps only
its subgroups and its classes, structs and concepts, without their members;
a group that has subgroups keeps only its subgroups. The pages still
document every member.
"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "build-docs/html")
kind = re.compile(r"^(group__|struct|class|concept|union)[^#]*$")
n_files = n_removed = 0


def count(items):
    return sum(1 + (count(c) if isinstance(c, list) else 0) for _, _, c in items)


for p in sorted(out.glob("group__*.js")):
    text = p.read_text(encoding="utf-8")
    head, _, body = text.partition("=")
    items = json.loads(body.strip().rstrip(";"))
    # a group with subgroups shows only its subgroups: its own types stay on its page
    def is_group(href):
        return bool(href) and href.startswith("group__") and "#" not in href

    has_subgroups = any(is_group(href) for _, href, _ in items)
    kept = []
    for title, href, child in items:
        if has_subgroups and not is_group(href):
            continue
        if href and kind.match(href):
            # a subgroup's child is the name of its own file; a type's members are dropped
            kept.append([title, href, child if isinstance(child, str) else None])
    if count(kept) == count(items):
        continue
    n_removed += count(items) - count(kept)
    n_files += 1
    p.write_text(head + "=\n" + json.dumps(kept, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
print(f"trim_navtree: {n_removed} entries removed from {n_files} group trees")
