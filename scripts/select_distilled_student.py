"""Select by retrieval metrics, retaining the public initialization on a tie."""
import argparse
from pathlib import Path

from foliorecall.io import read_json, write_json, provenance

parser = argparse.ArgumentParser()
parser.add_argument("--training", required=True)
parser.add_argument("--initialization", default="outputs/stage3/public-ml-cuda")
parser.add_argument("--output", required=True)
args = parser.parse_args()
training, output = Path(args.training), Path(args.output)
if output.exists() and any(output.iterdir()):
    raise ValueError("选模输出目录非空")
initial = read_json(Path(args.initialization) / "result.json")
initial_run = read_json(Path(args.initialization) / "run.json")
candidates = [{"name": "public-initialization", "step": 0, "metrics": initial["metrics"],
    "result": str(Path(args.initialization) / "result.json"),
    "query_config": initial_run["config"]["query_config"]}]
for row in sorted(read_json(training / "checkpoint-evaluations.json"), key=lambda r: r["step"]):
    folder = Path(row["checkpoint"])
    candidates.append(dict(row, name=f"step-{row['step']}",
        result=str(folder / "dev-evaluation" / "result.json"),
        query_config=read_json(folder / "query-config.json")))
if len(candidates) < 2:
    raise ValueError("尚无已评测训练检查点")
key = lambda row: tuple(row["metrics"][k] for k in ("nDCG@10", "Recall@10", "Recall@5"))
best = max(candidates, key=key)
trained = max(candidates[1:], key=key)
delta = {k: trained["metrics"][k]-initial["metrics"][k] for k in initial["metrics"]}
trained_result = read_json(trained["result"])
baseline_rows = {r["query_id"]: r for r in initial["results"]}
if {r["query_id"] for r in trained_result["results"]} != baseline_rows.keys():
    raise ValueError("选模开发查询不一致")
changes = []
for row in trained_result["results"]:
    before = baseline_rows[row["query_id"]]
    if row["query"] != before["query"]:
        raise ValueError("选模开发查询文本不一致")
    changes.append({"query_id": row["query_id"], "query": row["query"],
        "delta_ndcg": row["nDCG@10"]-before["nDCG@10"], "initialization": before, "trained": row})
changes.sort(key=lambda r: (-abs(r["delta_ndcg"]), r["query_id"]))
provenance(output, {"training": str(training), "initialization": args.initialization})
write_json(output / "result.json", {"candidates": candidates, "selected": best,
    "best_trained": trained, "best_trained_delta": delta,
    "expansion_quality_gate": delta["nDCG@10"] >= 0.001 and delta["Recall@10"] >= 0,
    "expansion_budget_gate": "must estimate separately before starting",
    "default_deployment": None, "rule": "nDCG@10, Recall@10, Recall@5; ties retain earlier candidate",
    "query_changes": {"improved": sum(r["delta_ndcg"] > 1e-12 for r in changes),
        "worse": sum(r["delta_ndcg"] < -1e-12 for r in changes),
        "unchanged": sum(abs(r["delta_ndcg"]) <= 1e-12 for r in changes)}})
write_json(output / "query-changes.json", changes)
write_json(output / "review-candidates.json", changes[:5])
print({"selected": best["name"], "best_trained": trained["name"], "delta": delta})
