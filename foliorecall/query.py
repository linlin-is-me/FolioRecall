"""Independent query encoders; page indexes retain their teacher identity."""
import json
from pathlib import Path
import time

import numpy as np

from .search import encoding_identity, search


def validate_query_config(config, teacher_config):
    if config.get("kind") != "nanovdr":
        raise ValueError("查询学生 kind 必须为 nanovdr")
    if config.get("target_encoding_identity") != encoding_identity(teacher_config):
        raise ValueError("学生目标教师身份与索引不匹配")
    expected = {"dimension": 2048, "pooling": "mean", "normalize": True,
                "max_length": 512, "prompt": "", "projection": [768, 768, 2048]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"学生表示配置不匹配: {key}")
    if not config.get("revision") or config.get("tokenizer_revision") != config["revision"]:
        raise ValueError("学生及 tokenizer 必须固定同一 revision")
    if config.get("tokenizer_id") != config.get("model_id"):
        raise ValueError("必须使用发布包配套 tokenizer")


def load_query_encoder(config):
    if config.get("kind") != "nanovdr":
        from .encoding import load_encoder
        return load_encoder(config)
    import torch
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；CPU 示例请显式使用 student-ml-cpu.json")
    from sentence_transformers import SentenceTransformer
    if config.get("model_path"):
        saved = json.loads((Path(config["model_path"]) / "query-config.json").read_text())
        for key in ("model_id", "revision", "tokenizer_id", "tokenizer_revision", "target_encoding_identity",
                    "dimension", "pooling", "projection", "normalize", "max_length", "prompt"):
            if saved.get(key) != config.get(key):
                raise ValueError(f"本地学生包与查询配置不匹配: {key}")
    model = SentenceTransformer(config.get("model_path") or config["model_id"],
        revision=None if config.get("model_path") else config["revision"],
        device=config["device"], trust_remote_code=False,
        model_kwargs={"dtype": getattr(torch, config["dtype"]), "attn_implementation": "sdpa"})
    names = [type(module).__name__ for module in model]
    if names != ["Transformer", "Pooling", "Dense", "Dense", "Normalize"]:
        raise ValueError(f"学生模块与发布结构不匹配: {names}")
    if model[1].pooling_mode != "mean" or model.get_embedding_dimension() != 2048:
        raise ValueError("学生 pooling 或维度不匹配")
    if [(model[i].linear.in_features, model[i].linear.out_features) for i in (2, 3)] != [(768, 768), (768, 2048)]:
        raise ValueError("学生投影层不匹配")
    if [type(model[i].activation_function).__name__ for i in (2, 3)] != ["GELU", "Identity"]:
        raise ValueError("学生投影激活不匹配")
    if model.max_seq_length != config["max_length"] or any(model.prompts.values()):
        raise ValueError("学生长度或 prompt 与发布配置不匹配")
    backbone = model[0].model.config
    if (backbone.model_type, backbone.n_layers, backbone.dim, backbone.vocab_size) != ("distilbert", 6, 768, 30522):
        raise ValueError("学生骨干与发布配置不匹配")
    model.eval()
    return model


def load_retriever(folder, teacher_config, query_config=None, candidate_pages=None):
    """Validate identities and the full corpus before loading query weights."""
    from .search import load_index
    if query_config is not None:
        validate_query_config(query_config, teacher_config)
    index, pages = load_index(folder, teacher_config)
    if candidate_pages is not None:
        from .evaluation import validate_candidate_corpus
        validate_candidate_corpus(pages, candidate_pages)
    if query_config is not None:
        import torch
        import faiss
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        faiss.omp_set_num_threads(1)
    model = load_query_encoder(query_config or teacher_config)
    return model, index, pages


def encode_query_texts(model, texts, config):
    if config.get("kind") != "nanovdr":
        from .encoding import encode_queries
        return encode_queries(model, texts, config)
    vectors = model.encode(texts, batch_size=config.get("batch_size", 1),
        prompt="", normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), config["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("学生输出维度或数值异常")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("学生输出零向量")
    return vectors / norms


def retrieve(model, config, index, pages, text, top_k=5):
    def sync():
        if config["device"] == "cuda":
            import torch
            torch.cuda.synchronize()
    sync()
    started = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("查询不能为空")
    vectors = encode_query_texts(model, [text], config)
    sync()
    encoded = time.perf_counter()
    ranked = search(index, pages, vectors, top_k)[0]
    searched = time.perf_counter()
    payload = json.dumps(ranked, ensure_ascii=False)
    ended = time.perf_counter()
    return ranked, payload, {"encode_seconds": encoded-started,
        "query_seconds": searched-started, "request_seconds": ended-started}


def process_memory():
    """WSL/Linux RSS; process high-water mark includes model loading."""
    path = Path("/proc/self/status")
    if not path.exists():
        return {"rss_bytes": None, "peak_rss_bytes": None}
    values = {line.split(':')[0]: int(line.split()[1])*1024 for line in path.read_text().splitlines()
              if line.startswith(("VmRSS:", "VmHWM:"))}
    return {"rss_bytes": values.get("VmRSS"), "peak_rss_bytes": values.get("VmHWM")}


def weight_file_bytes(config):
    if config.get("model_path"):
        folder = Path(config["model_path"])
    else:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(config["model_id"], "config.json", revision=config["revision"])
        if not isinstance(cached, str):
            return None
        folder = Path(cached).parent
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()
               and (p.suffix == ".safetensors" or p.name == "pytorch_model.bin"))


def benchmark(model, config, index, pages, queries, qrels, warmups=5, repeats=3):
    import torch
    import faiss
    from .evaluation import metrics
    if not queries or warmups < 1 or repeats < 1:
        raise ValueError("查询、预热或重复次数无效")
    ids = [str(q["query_id"]) for q in queries]
    if len(ids) != len(set(ids)) or set(ids) != set(qrels):
        raise ValueError("开发查询与标签 ID 不一致或重复")
    candidates = {p["page_id"] for p in pages}
    for qid in ids:
        if not set(qrels[qid]).issubset(candidates):
            raise ValueError("候选库缺失 qrels 页面")
    for i in range(warmups):
        retrieve(model, config, index, pages, queries[i % len(queries)]["query"], 10)
    warm_memory = process_memory()
    if config["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    requests, results, rounds = [], [], []
    def percentiles(rows, key):
        return {f"p{p}": float(np.percentile([r[key] for r in rows], p)) for p in (50, 95)}
    for repeat in range(repeats):
        current = []
        for query in queries:
            ranked, _, times = retrieve(model, config, index, pages, query["query"], 10)
            row = dict(times, repeat=repeat, query_id=str(query["query_id"]))
            requests.append(row)
            current.append(row)
            if repeat == 0:
                results.append({"query_id": str(query["query_id"]), "query": query["query"],
                    "ranking": ranked, **metrics([p["page_id"] for p in ranked], qrels[str(query["query_id"])])})
            elif [p["page_id"] for p in ranked] != [p["page_id"] for p in results[len(current)-1]["ranking"]]:
                raise ValueError("重复测量排名变化，需排查确定性")
        rounds.append({key: percentiles(current, key) for key in times})
        print(f"benchmark repeat {repeat+1}/{repeats} complete", flush=True)
    return {"scope": "fixed internal development; upstream student VDR overlap unverified",
        "queries": len(queries), "candidates": index.ntotal, "warmups": warmups, "repeats": repeats,
        "metrics": {key: float(np.mean([r[key] for r in results])) for key in ("nDCG@10", "Recall@5", "Recall@10")},
        **{key: percentiles(requests, key) for key in times}, "rounds": rounds,
        "warm_memory": warm_memory, "final_memory": process_memory(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if config["device"] == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "faiss_threads": faiss.omp_get_max_threads(),
        "cpu": next((line.split(':', 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
        "weight_file_bytes": weight_file_bytes(config),
        "dtype": config["dtype"], "device": config["device"], "batch_size": 1,
        "timing_scope": "resident text input to result JSON; encode includes tokenization and CUDA synchronization; excludes loading, transport, image rendering and metric calculation",
        "results": results, "requests": requests}
