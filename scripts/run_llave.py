"""Prepare one LLaVE arm and compare it with preserved teacher baselines.

GPU work uses the existing train/index/evaluate CLI, not a second executor.
"""
import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path

from foliorecall.io import provenance, read_json, read_rows, write_json


ROOT = Path("outputs/extensions/B/llave-v1")
HISTORICAL = Path("outputs/stage2/negative-control/filtered")
DATA = Path("data/vdr-stage2-control/filtered")
DEV = Path("data/vdr-stage2/dev")
METRICS = ("nDCG@10", "Recall@5", "Recall@10")


def prepare(root):
    from foliorecall.trainer import batch_plan, training_dataset, validate_training_data

    if (root / "preparation.json").exists():
        raise ValueError("准备记录已存在；保留原件并使用现有配置接续")
    config = read_json("configs/lora-baseline.json")
    historical = read_json(HISTORICAL / "settings.json")
    frozen = read_json("outputs/extensions/B/preparation-v1/llave-batches.json")
    split, pages = read_json(DATA / "split.json"), read_rows(DATA / "pages.jsonl")
    mapping = validate_training_data(split, pages, read_rows(DEV / "pages.jsonl"))
    batches = batch_plan(training_dataset(split["train"], mapping), 4, 42)
    if config != historical["config"] or split["train"] != historical["rows"] or frozen["rows"] != historical["rows"]:
        raise ValueError("历史普通基线与本轮配置或训练样本不一致，不能直接复用")
    if batches != historical["batches"] or batches != frozen["batches"] or len(batches) != 64:
        raise ValueError("训练批次与冻结64批不一致")
    if len(split["train"]) != 256 or any(len(b) != 4 for b in batches):
        raise ValueError("需要256条训练样本和64个完整batch")
    baseline = read_json(HISTORICAL / "result.json")
    if not baseline["complete"] or baseline["completed_steps"] != 64:
        raise ValueError("历史普通基线未完成64步")
    for path in (HISTORICAL / "checkpoint-64/encoding.json",
                 HISTORICAL / "checkpoint-64/dev-index/config.json",
                 HISTORICAL / "checkpoint-64/dev-evaluation/result.json"):
        read_json(path)
    candidate = deepcopy(config)
    candidate["train"].update(hardness_mode="in_batch_negatives", hardness_strength=9)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "llave-config.json", candidate)
    write_json(root / "batches.json", {"rows": split["train"], "batches": batches})
    packages = {name: importlib.metadata.version(name) for name in (
        "torch", "sentence-transformers", "transformers", "peft", "accelerate", "datasets")}
    result = {"status": "prepared", "candidate_config": str(root / "llave-config.json"),
        "historical_baseline": str(HISTORICAL), "historical_commit": "21c2ee97beaec23899e7813c6b000d3f095c2ddb",
        "historical_baseline_reused": True, "training_rows_match": True, "batch_order_match": True,
        "configuration_matches_except_loss": True, "training_queries": 256, "steps": 64,
        "development_queries": 200, "development_pages": 1000, "packages": packages,
        "selection": "last step 64 only", "new_api_calls": 0,
        "old_B0_protocol": "preserved; separate four-step run and fixed +0.005 expansion gate superseded",
        "limits": ["single seed, previously seen development set", "three negative replacements are assistant-reviewed",
                   "historical resource figures are not a same-run performance comparison",
                   "new teacher is not compatible with the existing public query student by assumption"]}
    provenance(root / "preparation-run", result)
    write_json(root / "preparation.json", result)
    return result


def paired(reference, candidate):
    old = {r["query_id"]: r for r in reference["results"]}
    new = {r["query_id"]: r for r in candidate["results"]}
    if len(old) != 200 or old.keys() != new.keys() or len(candidate["results"]) != 200:
        raise ValueError("评测必须包含相同200个唯一查询")
    rows = []
    for qid, a in old.items():
        b = new[qid]
        if a["query"] != b["query"]:
            raise ValueError("配对查询原文不一致")
        delta = {k: b[k] - a[k] for k in METRICS}
        rows.append({"query_id": qid, "query": a["query"], "delta": delta,
            "baseline": {k: a[k] for k in METRICS}, "candidate": {k: b[k] for k in METRICS},
            "baseline_ranking": [p["page_id"] for p in a["ranking"]],
            "candidate_ranking": [p["page_id"] for p in b["ranking"]]})
    summary = {k: {"mean_delta": sum(r["delta"][k] for r in rows) / len(rows),
        "improved": sum(r["delta"][k] > 1e-12 for r in rows),
        "regressed": sum(r["delta"][k] < -1e-12 for r in rows),
        "tied": sum(abs(r["delta"][k]) <= 1e-12 for r in rows)} for k in METRICS}
    changed = [r for r in rows if r["baseline_ranking"] != r["candidate_ranking"]]
    summary.update(ranking_changed=len(changed), ranking_changed_metrics_tied=sum(
        all(abs(r["delta"][k]) <= 1e-12 for k in METRICS) for r in changed))
    return summary, rows


def report(root):
    read_json(root / "preparation.json")
    candidate = read_json(root / "evaluation/result.json")
    ordinary = read_json(HISTORICAL / "checkpoint-64/dev-evaluation/result.json")
    original = read_json("outputs/stage2/dev-original/result.json")
    for item in (candidate, ordinary, original):
        if item["candidates"] != 1000 or item["queries"] != 200:
            raise ValueError("需要完整1000页、200题开发结果")
    training = read_json(root / "training/result.json")
    if not training["complete"] or training["completed_steps"] != 64:
        raise ValueError("LLaVE训练未完成64步")
    if training["loss"] != {"name": "CachedMultipleNegativesRankingLoss", "scale": 20.0,
                            "hardness_mode": "in_batch_negatives", "hardness_strength": 9.0}:
        raise ValueError("候选损失设置不符")
    comparisons = {}
    for name, reference in (("ordinary", ordinary), ("original", original)):
        summary, rows = paired(reference, candidate)
        comparisons[name] = summary
        write_json(root / f"paired-{name}.json", rows)
        if name == "ordinary":
            improvements = sorted((r for r in rows if r["delta"]["nDCG@10"] > 1e-12),
                                  key=lambda r: -r["delta"]["nDCG@10"])[:5]
            regressions = sorted((r for r in rows if r["delta"]["nDCG@10"] < -1e-12),
                                 key=lambda r: r["delta"]["nDCG@10"])[:5]
            write_json(root / "cases-to-review.json", {"improvements": improvements, "regressions": regressions})
    build = read_json(root / "dev-index/build.json")
    result = {"status": "measured_pending_case_review", "queries": 200, "pages": 1000,
        "metrics": {"original_historical": original["metrics"], "ordinary_historical": ordinary["metrics"],
                    "llave": candidate["metrics"]}, "paired": comparisons,
        "resources": {"llave_training": {k: v for k, v in training.items() if k not in ("steps", "log_history")},
            "llave_build": build, "llave_query_seconds": candidate["query_seconds"],
            "llave_evaluation_peak_cuda_bytes": candidate["peak_cuda_bytes"],
            "historical_training_seconds": read_json(HISTORICAL / "result.json")["segment_seconds"],
            "historical_build_seconds": read_json(HISTORICAL / "checkpoint-64/dev-index/build.json")["seconds"]},
        "resource_comparison": "historical baseline; no strict same-run speed/memory claims",
        "new_api_calls": 0, "new_api_cost_cny": 0, "product_teacher_changed": False,
        "scope": "single-seed, seen internal development; not independent acceptance"}
    write_json(root / "comparison.json", result)
    return {"metrics": result["metrics"], "paired": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    import json
    print(json.dumps((prepare if args.command == "prepare" else report)(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
