"""Query-only teacher targets, bound to immutable IDs, text and encoding settings."""
from collections import Counter
from pathlib import Path
import random
import time

import numpy as np

from .data_preparation import normalized_query
from .io import read_json, read_rows, write_json, write_rows, provenance
from .search import encoding_identity


def training_queries(data, count=3000, metadata="data/vdr/english-metadata.jsonl"):
    split = read_json(Path(data) / "split.json")
    original = [{"query_id": r["query_id"], "query": r["query"]} for r in split["train"]]
    if count < 1:
        raise ValueError("训练查询数量必须为正")
    rows = original[:count]
    if count > len(original):
        # Reconstruct the original full page partition; never split only the 200 dev queries.
        source = read_json(Path(data) / "source.json")
        all_rows = read_rows(metadata)
        ids = sorted(r["id"] for r in all_rows)
        if len(ids) != len(set(ids)):
            raise ValueError("元数据页面 ID 重复")
        random.Random(source["seed"]).shuffle(ids)
        dev_pool = set(ids[:len(ids)//5])
        duplicates = Counter(normalized_query(r.get("query")) for r in all_rows)
        existing = {r["query_id"] for r in original}
        if existing & dev_pool:
            raise ValueError("原训练查询与重建页面切分不一致")
        eligible = [{"query_id": r["id"], "query": r["query"]} for r in all_rows
            if r["id"] not in dev_pool | existing and normalized_query(r.get("query"))
            and duplicates[normalized_query(r["query"])] == 1]
        eligible.sort(key=lambda r: r["query_id"])
        if len(eligible) < count-len(rows):
            raise ValueError("训练侧可扩充查询不足")
        rows += random.Random(source["seed"]).sample(eligible, count-len(rows))
    ids = [r["query_id"] for r in rows]
    texts = [normalized_query(r["query"]) for r in rows]
    held_out = {normalized_query(r["query"]) for r in split["dev"]}
    held_ids = {r["query_id"] for r in split["dev"]}
    dev_pages = Path(data) / "dev" / "pages.jsonl"
    if dev_pages.exists():
        held_ids.update(r["page_id"] for r in read_rows(dev_pages))
    if (len(set(ids)) != len(ids) or len(set(texts)) != len(texts) or "" in texts
            or set(texts) & held_out or set(ids) & held_ids):
        raise ValueError("训练查询重复、为空或与开发集交叉")
    return rows


def validate_vectors(vectors, count, dimension):
    if vectors.dtype != np.float32 or vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError("目标向量维度、类型或数值无效")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("教师目标必须为单位向量")


def read_chunk(path, expected, dimension):
    with np.load(path, allow_pickle=False) as chunk:
        if (chunk["query_ids"].tolist() != [r["query_id"] for r in expected]
                or chunk["texts"].tolist() != [r["query"] for r in expected]):
            raise ValueError("缓存目标与查询 ID、文本或顺序不匹配")
        vectors = chunk["vectors"].copy()
    validate_vectors(vectors, len(expected), dimension)
    return vectors


def load_targets(folder, query_config, limit=None):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest["encoding_identity"] != query_config["target_encoding_identity"]:
        raise ValueError("目标缓存教师与学生配置不匹配")
    rows = read_rows(folder / "queries.jsonl")
    if len(rows) != manifest["count"]:
        raise ValueError("缓存查询清单数量变化")
    limit = len(rows) if limit is None else limit
    if not 0 < limit <= len(rows):
        raise ValueError("请求的目标数量无效")
    chunks = []
    for start in range(0, limit, manifest["chunk_size"]):
        expected = rows[start:start+manifest["chunk_size"]]
        chunks.append(read_chunk(folder / "chunks" / f"{start:06d}.npz", expected, query_config["dimension"]))
    vectors = np.concatenate(chunks)[:limit]
    return rows[:limit], vectors, manifest


def cache_teacher(teacher_path, data, output, count=3000, stop_after=None, max_seconds=3600):
    teacher = read_json(teacher_path)
    if teacher.get("status") != "selected":
        raise ValueError("教师尚未选定")
    config = teacher["encoding_config"]
    if encoding_identity(config) != teacher["index_encoding_identity"]:
        raise ValueError("教师交接记录内部不匹配")
    rows = training_queries(data, count)
    output = Path(output)
    manifest = {"teacher": str(teacher_path), "encoding_config": config,
        "encoding_identity": encoding_identity(config), "data": str(Path(data).resolve()),
        "count": len(rows), "chunk_size": 64,
        "split_scope": "existing page partition and normalized-query exclusion; upstream overlap unverified"}
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json") != manifest or read_rows(output / "queries.jsonl") != rows:
            raise ValueError("已有目标清单或教师配置不匹配；使用新目录")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("目标输出目录非空且没有完整清单")
        write_json(output / "manifest.json", manifest)
        write_rows(output / "queries.jsonl", rows)
    run = output / "runs" / str(time.time_ns())
    provenance(run, dict(manifest, stop_after=stop_after, max_seconds=max_seconds))
    (output / "chunks").mkdir(exist_ok=True)
    started = time.monotonic()
    model, completed, generated = None, 0, 0
    try:
        for start in range(0, len(rows), manifest["chunk_size"]):
            expected = rows[start:start+manifest["chunk_size"]]
            chunk_path = output / "chunks" / f"{start:06d}.npz"
            if chunk_path.exists():
                read_chunk(chunk_path, expected, config["dimension"])
                completed += len(expected)
                continue
            if (stop_after is not None and generated >= stop_after) or time.monotonic()-started >= max_seconds:
                break
            if model is None:
                from .encoding import load_encoder, encode_queries
                model = load_encoder(config)
            vectors = encode_queries(model, [r["query"] for r in expected], config)
            validate_vectors(vectors, len(expected), config["dimension"])
            temporary = chunk_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, query_ids=np.asarray([r["query_id"] for r in expected]),
                    texts=np.asarray([r["query"] for r in expected]), vectors=vectors)
            temporary.replace(chunk_path)
            completed += len(expected)
            generated += len(expected)
            write_json(output / "status.json", {"completed": completed, "count": count, "complete": completed == count,
                "generated_this_run": generated, "seconds": time.monotonic()-started})
            print(f"teacher targets {completed}/{count}; {time.monotonic()-started:.1f}s", flush=True)
    except Exception as exc:
        write_json(run / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    result = {"completed": completed, "count": count, "complete": completed == count,
        "generated_this_run": generated, "seconds": time.monotonic()-started}
    write_json(output / "status.json", result)
    write_json(run / "result.json", result)
    return result
