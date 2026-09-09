import math
import time
import numpy as np


def metrics(ranked_ids, relevance):
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("排名包含重复页面")
    positives = {str(key): float(value) for key, value in relevance.items() if value > 0}
    if not positives:
        raise ValueError("查询没有正相关标签")
    def dcg(gains):
        # Linear graded gain, matching standard trec_eval nDCG.
        return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = dcg(sorted(positives.values(), reverse=True)[:10])
    return {"nDCG@10": dcg([positives.get(str(x), 0) for x in ranked_ids[:10]]) / ideal,
            **{f"Recall@{k}": len(set(map(str, ranked_ids[:k])) & positives.keys()) / len(positives) for k in (5, 10)}}


def evaluate(model, config, index, pages, queries, qrels):
    import torch
    from .encoding import encode_queries
    from .search import search
    candidate_ids = {p["page_id"] for p in pages}
    results, timings, encode_times = [], [], []
    # Warmup is separate from timed queries.
    encode_queries(model, [queries[0]["query"]], config)
    if config["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for query in queries:
        query_id = str(query["query_id"])
        relevance = qrels[query_id]
        if not set(relevance).issubset(candidate_ids):
            raise ValueError("候选库缺失 qrels 页面")
        start = time.perf_counter()
        vector = encode_queries(model, [query["query"]], config)
        encoded = time.perf_counter()
        ranked = search(index, pages, vector, 10)[0]
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        encode_times.append(encoded - start)
        results.append({"query_id": query_id, "query": query["query"], "ranking": ranked,
                        **metrics([p["page_id"] for p in ranked], relevance)})
    return {"scope": "HR English 20-query flow check; not full benchmark", "queries": len(results),
            "candidates": index.ntotal,
            "metrics": {key: float(np.mean([row[key] for row in results])) for key in ("nDCG@10", "Recall@5", "Recall@10")},
            "query_seconds": {f"p{p}": float(np.percentile(timings, p)) for p in (50, 95)},
            "encode_seconds": {f"p{p}": float(np.percentile(encode_times, p)) for p in (50, 95)},
            "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None,
            "results": results}
