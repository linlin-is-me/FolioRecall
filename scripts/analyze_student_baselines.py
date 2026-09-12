"""Compare saved student rankings without loading any model or changing labels."""
import argparse
import json
from pathlib import Path

from foliorecall.evaluation import metrics
from foliorecall.io import read_json, write_json, provenance

parser = argparse.ArgumentParser()
parser.add_argument("--names", nargs="+", default=["public-en-cpu", "public-ml-cpu"])
parser.add_argument("--output", required=True)
parser.add_argument("--teacher-result", help="Use a newly measured teacher instead of historical quality only")
args = parser.parse_args()
output = Path(args.output)
if output.exists() and any(output.iterdir()):
    raise ValueError("分析输出目录非空")
teacher = read_json("outputs/stage2/teacher.json")
reference_path = args.teacher_result or teacher["candidates"]["original"]["result"]
reference = read_json(reference_path)
baseline = {r["query_id"]: r for r in reference["results"]}
qrels = read_json("data/vdr-stage2/dev/qrels.json")
reports, runs, query_changes = {}, {}, {}
for name in args.names:
    path = Path("outputs/stage3") / name
    result = read_json(path / "result.json")
    rows = {r["query_id"]: r for r in result["results"]}
    if rows.keys() != baseline.keys() or result["candidates"] != 1000:
        raise ValueError("比较任务不匹配")
    changes = []
    for qid, row in rows.items():
        if row["query"] != baseline[qid]["query"]:
            raise ValueError("开发查询文本变化")
        recalculated = metrics([p["page_id"] for p in row["ranking"]], qrels[qid])
        if any(abs(recalculated[k]-row[k]) > 1e-12 for k in recalculated):
            raise ValueError("保存指标与当前标签不匹配")
        delta = row["nDCG@10"]-baseline[qid]["nDCG@10"]
        changes.append(delta)
        query_changes.setdefault(qid, {"query_id": qid, "query": row["query"],
            "reference": baseline[qid], "students": {}})["students"][name] = {"delta_ndcg": delta, "result": row}
    reports[name] = {key: result[key] for key in ("metrics", "encode_seconds", "query_seconds", "request_seconds",
        "warm_memory", "final_memory", "peak_cuda_bytes", "peak_cuda_reserved_bytes", "dtype", "device",
        "weight_file_bytes", "parameter_count", "model_loading_seconds", "index_bytes", "torch_threads")}
    reports[name]["delta_from_teacher"] = {k: result["metrics"][k]-reference["metrics"][k] for k in result["metrics"]}
    reports[name]["queries"] = {"improved": sum(x > 1e-12 for x in changes),
        "worse": sum(x < -1e-12 for x in changes), "unchanged": sum(abs(x) <= 1e-12 for x in changes)}
    if args.teacher_result and result["device"] == "cuda":
        conditions = ("device", "dtype", "gpu", "batch_size", "torch_threads", "faiss_threads", "warmups", "repeats")
        if any(result[k] != reference[k] for k in conditions):
            raise ValueError("GPU加速比测量条件不同")
        reports[name]["gpu_encode_speedup_p50"] = reference["encode_seconds"]["p50"] / result["encode_seconds"]["p50"]
        reports[name]["gpu_request_speedup_p50"] = reference["request_seconds"]["p50"] / result["request_seconds"]["p50"]
    runs[name] = read_json(path / "run.json")
cases = sorted(query_changes.values(), key=lambda row: (-max(abs(s["delta_ndcg"]) for s in row["students"].values()), row["query_id"]))[:5]
result = {"teacher": "outputs/stage2/teacher.json", "teacher_metrics": reference["metrics"], "students": reports,
    "comparison_scope": "same frozen 200 English queries and 1000 original-teacher pages; no labels modified",
    "teacher_result": reference_path,
    "timing_scope": "CPU/GPU deployment rows separate; GPU speedup only for a supplied teacher with matching measurement conditions",
    "initialization": "pending both GPU BF16 public baselines", "default_deployment": "pending user choice",
    "limitations": ["single near-ceiling internal task", "upstream public-student VDR overlap unverified",
        "upstream teacher query instruction differs", "original-document isolation unverified"]}
public_gpu = [name for name in ("public-en-cuda", "public-ml-cuda") if name in reports]
if len(public_gpu) == 2 and all(reports[name]["device"] == "cuda" for name in public_gpu):
    winner = max(public_gpu, key=lambda name: tuple(reports[name]["metrics"][k] for k in ("nDCG@10", "Recall@10", "Recall@5")) + (int("-ml-" in name),))
    result["initialization"] = {"name": winner, "query_config": runs[winner]["config"]["query_config"],
        "rule": "nDCG@10, Recall@10, Recall@5, then ML on exact tie; not deployment selection"}
provenance(output, {"names": args.names, "source_runs": runs})
write_json(output / "result.json", result)
write_json(output / "review-candidates.json", cases)
print(json.dumps(result, indent=2, ensure_ascii=False))
