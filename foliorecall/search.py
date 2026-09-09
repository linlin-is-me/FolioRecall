from pathlib import Path
import faiss
import numpy as np

from .documents import validate_pages
from .io import read_json, read_rows, write_json, write_rows


def encoding_identity(config):
    return {key: value for key, value in config.items() if key not in {"train", "seed", "batch_size", "device"}}


def save_index(vectors, rows, config, output):
    validate_pages(rows)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if not len(rows) or vectors.shape != (len(rows), config["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("空索引或页面与向量不匹配")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError("索引向量必须归一化")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(config["dimension"])
    index.add(vectors)
    faiss.write_index(index, str(output / "index.faiss"))
    write_rows(output / "pages.jsonl", rows)
    write_json(output / "config.json", config)
    return index


def load_index(folder, config):
    folder = Path(folder)
    if encoding_identity(read_json(folder / "config.json")) != encoding_identity(config):
        raise ValueError("查询编码配置与索引不匹配，请使用同一模型、适配器与预处理配置")
    rows = read_rows(folder / "pages.jsonl")
    index = faiss.read_index(str(folder / "index.faiss"))
    if not rows or index.ntotal != len(rows) or index.d != config["dimension"]:
        raise ValueError("空索引或索引元数据不匹配")
    return index, rows


def search(index, rows, query_vectors, top_k=5):
    if top_k < 1 or index.ntotal == 0:
        raise ValueError("top_k 必须为正，索引不能为空")
    scores, positions = index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), min(top_k, len(rows)))
    return [[dict(rows[int(pos)], score=float(score)) for score, pos in zip(qscores, qpositions)]
            for qscores, qpositions in zip(scores, positions)]
