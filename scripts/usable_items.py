"""Count usable verified items: the frozen main set, and passed eligible items after it (the replication pool).

Usage: uv run python scripts/usable_items.py <manifest> <screen dir> <decisions.jsonl>. No GPU, no model outcomes.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from kvsel.confirm import load_flags, main_set
from kvsel.verify import PASS, load_decisions

manifest, screen, decisions_path = sys.argv[1:4]
items = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
flags = load_flags(screen)
decisions, stale = load_decisions(decisions_path, items)
main = main_set(items, screen, decisions_path)["main"]
cut = [i["item_id"] for i in items].index(main[-1])
after = [i["item_id"] for i in items[cut + 1:] if flags.get(i["item_id"])]
state = Counter("unverified" if i not in decisions else "passed" if decisions[i]["decision"] == PASS else "failed" for i in after)
first_gap = next((n for n, i in enumerate(after) if i not in decisions), len(after))
print(json.dumps({
    "manifest_items": len(items), "screened": len(flags), "eligible": sum(flags.values()),
    "decisions_current": len(decisions), "decisions_stale": stale,
    "passed_total": sum(d["decision"] == PASS for d in decisions.values()),
    "main_set": len(main),
    "after_main_eligible": len(after), "after_main": dict(state),
    "after_main_passed_before_first_unverified": sum(decisions[i]["decision"] == PASS for i in after[:first_gap]),
    "first_unverified_after_main": after[first_gap] if first_gap < len(after) else None,
}, indent=2))
