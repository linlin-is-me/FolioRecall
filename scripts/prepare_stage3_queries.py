"""Validate query-only preparation; no encoder import, targets or training."""
from pathlib import Path
from foliorecall.io import write_json, write_rows, provenance
from foliorecall.targets import training_queries

root = Path("outputs/stage3/query-preparation")
if root.exists() and any(root.iterdir()):
    raise ValueError("已有查询准备记录；不覆盖")
rows = training_queries("data/vdr-stage2", 3000)
expanded = training_queries("data/vdr-stage2", 10000)
assert expanded[:3000] == rows
result = {"training_queries": len(rows), "extension_feasibility_queries": len(expanded),
    "original3000_preserved": True, "targets_generated": False, "expansion_started": False,
    "selection": "existing train split; extension reconstructs full seed42 page partition and excludes normalized duplicate groups",
    "next": "await user notice before GPU baselines and teacher target generation"}
provenance(root, result)
write_rows(root / "queries.jsonl", rows)
write_json(root / "result.json", result)
print(result)
