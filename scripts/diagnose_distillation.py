"""Inspect alignment and coverage; development vectors stay outside training caches."""
import argparse
import numpy as np
from pathlib import Path

from foliorecall.io import read_json, read_rows, write_json, provenance
from foliorecall.query import load_query_encoder, encode_query_texts, validate_query_config
from foliorecall.targets import load_targets

parser = argparse.ArgumentParser()
parser.add_argument("role", choices=["teacher", "student"])
parser.add_argument("--query-config")
parser.add_argument("--teacher-diagnostic", default="outputs/stage3/alignment-teacher")
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists() and any(output.iterdir()):
    raise ValueError("诊断输出目录非空")
teacher = read_json("outputs/stage2/teacher.json")["encoding_config"]
dev = read_rows("data/vdr-stage2/dev/queries.jsonl")
config = teacher if args.role == "teacher" else read_json(args.query_config)
if args.role == "student":
    validate_query_config(config, teacher)
provenance(output, {"teacher": teacher, "query_config": config,
    "scope": "diagnostic only; development targets never enter training cache"})
import torch
torch.set_num_threads(4)
model = load_query_encoder(config)
dev_vectors = encode_query_texts(model, [r["query"] for r in dev], config)
np.savez(output / "dev-vectors.npz", query_ids=[r["query_id"] for r in dev], vectors=dev_vectors)
if args.role == "teacher":
    write_json(output / "result.json", {"queries": len(dev), "finite": bool(np.isfinite(dev_vectors).all()),
        "max_norm_error": float(np.abs(np.linalg.norm(dev_vectors, axis=1)-1).max())})
else:
    train, targets, manifest = load_targets("outputs/stage3/targets-3000", config, limit=64)
    train_vectors = encode_query_texts(model, [r["query"] for r in train], config)
    with np.load(Path(args.teacher_diagnostic) / "dev-vectors.npz", allow_pickle=False) as source:
        if source["query_ids"].tolist() != [r["query_id"] for r in dev]:
            raise ValueError("诊断查询顺序不一致")
        expected = source["vectors"].copy()
    summary = {}
    for name, student, reference in (("train_first64", train_vectors, targets), ("development200", dev_vectors, expected)):
        cosine = (student * reference).sum(axis=1)
        summary[name] = {"mean_cosine": float(cosine.mean()), "p5": float(np.percentile(cosine, 5)),
            "p50": float(np.median(cosine)), "max_norm_error": float(np.abs(np.linalg.norm(student, axis=1)-1).max())}
    full_train = read_rows("outputs/stage3/targets-3000/queries.jsonl")
    lengths = {}
    for name, rows in (("training3000", full_train), ("development200", dev)):
        encoded = model.tokenizer([r["query"] for r in rows], truncation=False, add_special_tokens=True)["input_ids"]
        counts = [len(ids) for ids in encoded]
        lengths[name] = {"count": len(rows), "max_tokens": max(counts),
            "p50_tokens": float(np.median(counts)), "p95_tokens": float(np.percentile(counts, 95)),
            "exceed_512": sum(n > 512 for n in counts)}
    from foliorecall.data_preparation import normalized_query
    overlap = {normalized_query(r["query"]) for r in full_train} & {normalized_query(r["query"]) for r in dev}
    write_json(output / "result.json", {"alignment": summary, "token_lengths": lengths,
        "normalized_train_dev_overlap": len(overlap), "teacher_identity_matches": True,
        "student_prompt": config["prompt"], "student_max_length": model.max_seq_length,
        "coverage_limit": "3000 queries from existing page split; document isolation and upstream training overlap unverified"})
    print(summary)
