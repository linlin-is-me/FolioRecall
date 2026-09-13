import argparse
import json
from pathlib import Path
import sys
import time

from .io import read_json, read_rows, write_json, provenance


def main():
    command_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="FolioRecall visual document retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    text_eval = sub.add_parser("evaluate-text", help="上游页面文本的固定 BM25 对照")
    text_eval.add_argument("--config", required=True)
    text_eval.add_argument("--data", required=True)
    text_eval.add_argument("--output", required=True)
    serve = sub.add_parser("serve", help="启动单模型常驻的本地页面检索演示")
    serve.add_argument("--index", required=True)
    serve.add_argument("--config", required=True)
    serve.add_argument("--query-config", required=True)
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--output", help="可选的应用验证记录目录，不记录到已有实验中")
    imp = sub.add_parser("import")
    imp.add_argument("inputs", nargs="*")
    imp.add_argument("--image-manifest")
    imp.add_argument("--output", required=True)
    imp.add_argument("--dpi", type=int, default=150)
    cache = sub.add_parser("cache-teacher")
    cache.add_argument("--teacher", default="outputs/stage2/teacher.json")
    cache.add_argument("--data", default="data/vdr-stage2")
    cache.add_argument("--output", required=True)
    cache.add_argument("--count", type=int, default=3000)
    cache.add_argument("--stop-after", type=int)
    cache.add_argument("--max-seconds", type=float, default=3600)
    student_train = sub.add_parser("distill")
    student_train.add_argument("--query-config", required=True)
    student_train.add_argument("--targets", required=True)
    student_train.add_argument("--output", required=True)
    student_train.add_argument("--index", default="indexes/vdr-dev-original")
    student_train.add_argument("--data", default="data/vdr-stage2/dev")
    student_train.add_argument("--limit", type=int)
    student_train.add_argument("--max-steps", type=int)
    student_train.add_argument("--micro-batch", type=int, default=32)
    student_train.add_argument("--max-seconds", type=float, default=3600)
    student_train.add_argument("--resume")
    student_train.add_argument("--skip-evaluation", action="store_true", help="Only for the initial four-step integration probe")
    for name in ("index", "query", "evaluate", "train-smoke", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/lora-baseline.json" if name == "train" else "configs/baseline.json")
        if name == "index":
            command.add_argument("--pages", required=True)
            command.add_argument("--output", required=True)
        elif name in ("query", "evaluate"):
            command.add_argument("--index", required=True)
            command.add_argument("--query-config")
            if name == "query":
                command.add_argument("text")
                command.add_argument("--top-k", type=int, default=5)
                command.add_argument("--json", action="store_true")
            else:
                command.add_argument("--data", default="data/hr")
                command.add_argument("--output", required=True)
                command.add_argument("--benchmark", action="store_true")
        else:
            command.add_argument("--data", default="data/vdr-stage2" if name == "train" else "data/vdr")
            command.add_argument("--output", required=True)
            command.add_argument("--gradient-checkpointing", action="store_true")
            if name == "train":
                command.add_argument("--resume")
                command.add_argument("--profile", action="store_true")
                command.add_argument("--stop-after", type=int)
                command.add_argument("--max-seconds", type=float, default=3600)
                command.add_argument("--checkpoint-only", action="store_true", help="Save checkpoints without development evaluation or final encoding probes")
    args = parser.parse_args()
    if args.command == "evaluate-text":
        from .text_baseline import evaluate_text
        result = evaluate_text(read_json(args.config), args.data, args.output)
        print(json.dumps(result['metrics'], indent=2))
        return 0
    if args.command == "serve":
        from .demo import serve_demo
        serve_demo(args.index, read_json(args.config), read_json(args.query_config), args.port, args.output)
        return 0
    if args.command == "cache-teacher":
        from .targets import cache_teacher
        print(json.dumps(cache_teacher(args.teacher, args.data, args.output, args.count, args.stop_after, args.max_seconds), indent=2))
        return 0
    if args.command == "distill":
        from .distillation import distill
        result = distill(read_json(args.query_config), args.targets, args.output, args.index, args.data,
            args.limit, args.max_steps, args.micro_batch, args.max_seconds, args.resume, not args.skip_evaluation)
        print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))
        return 0
    if args.command == "import":
        from .documents import import_documents
        result = import_documents(args.inputs, args.output, args.dpi, args.image_manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["pages"] else 1
    config = read_json(args.config)
    query_config = read_json(args.query_config) if getattr(args, "query_config", None) else config
    if args.command == "query":
        from .query import load_retriever, retrieve
        model, index, pages = load_retriever(args.index, config,
            query_config if args.query_config else None)
        result, payload, _ = retrieve(model, query_config, index, pages, args.text, args.top_k)
        if args.json:
            print(payload)
        else:
            for rank, row in enumerate(result, 1):
                print(f"{rank}. {row.get('document_name', row['doc_id'])} | page {row['page_number']} | {row['score']:.6f}")
                print(f"   source: {row['source']}\n   preview: {row['preview']}")
        return 0
    if args.command == "train":
        from .trainer import train
        if args.gradient_checkpointing:
            config["train"]["gradient_checkpointing"] = True
        try:
            train(config, args.data, args.output, args.resume, args.profile, args.stop_after, args.max_seconds, checkpoint_only=args.checkpoint_only)
        except Exception as exc:
            import torch
            failure = {"type": type(exc).__name__, "message": str(exc), "config": config,
                       "resume": args.resume, "profile": args.profile}
            if torch.cuda.is_available():
                failure.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                               reserved_cuda_bytes=torch.cuda.memory_reserved(),
                               total_cuda_bytes=torch.cuda.get_device_properties(0).total_memory)
            write_json(Path(args.output) / f"failure-{time.time_ns()}.json", failure)
            write_json(Path(args.output) / "failure.json", failure)
            raise
        return 0
    if args.command == "train-smoke":
        from .training import train_smoke
        try:
            train_smoke(config, args.data, args.output, args.gradient_checkpointing)
        except Exception as exc:
            write_json(Path(args.output) / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                       "gradient_checkpointing": args.gradient_checkpointing})
            raise
        return 0
    from .search import load_index, save_index, search
    if args.command in ("query", "evaluate"):
        if getattr(args, "query_config", None):
            from .query import validate_query_config
            validate_query_config(query_config, config)
        index, pages = load_index(args.index, config)
        if args.command == "evaluate":
            from .evaluation import validate_candidate_corpus
            validate_candidate_corpus(pages, read_rows(Path(args.data) / "pages.jsonl"))
    else:
        from .documents import validate_pages
        if (Path(args.output) / "index.faiss").exists():
            raise ValueError("索引已存在，请使用新输出目录")
        pages = validate_pages(read_rows(args.pages))
        if not pages:
            raise ValueError("页面清单为空")
    if args.command != "query":
        if args.command == "evaluate" and (Path(args.output) / "result.json").exists():
            raise ValueError("评测结果已存在，请使用新输出目录")
        provenance(args.output, {"page_config": config, "query_config": query_config} if getattr(args, "query_config", None) else config)
    loading_started = time.perf_counter()
    if args.command in ("query", "evaluate"):
        from .query import load_query_encoder
        if getattr(args, "benchmark", False) or getattr(args, "query_config", None):
            import torch
            import faiss
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            faiss.omp_set_num_threads(1)
        model = load_query_encoder(query_config)
    else:
        from .encoding import load_encoder, encode_pages
        model = load_encoder(config)
    loading_seconds = time.perf_counter() - loading_started
    if args.command == "index":
        import torch
        from PIL import Image
        from .encoding import processing
        from .query import process_memory
        memory_after_loading = process_memory()
        probe_started = time.perf_counter()
        with Image.open(pages[0]["preview"]) as image:
            features = model.preprocess([image.convert("RGB")], prompt=config["prompt"],
                                        processing_kwargs=processing(config))
        probe = {"first_page_id": pages[0]["page_id"], "max_seq_length": model.max_seq_length,
                 "pooling": model[1].pooling_mode, "input_ids_shape": list(features["input_ids"].shape),
                 "image_grid_thw": features["image_grid_thw"].tolist()}
        write_json(Path(args.output) / "encoding-probe.json", probe)
        del features
        if config["device"] == "cuda":
            torch.cuda.synchronize()
        probe_seconds = time.perf_counter() - probe_started
        start = time.perf_counter()
        vectors = encode_pages(model, pages, config)
        if config["device"] == "cuda":
            torch.cuda.synchronize()
        encoding_seconds = time.perf_counter() - start
        save_started = time.perf_counter()
        save_index(vectors, pages, config, args.output)
        result = {"pages": len(pages), "seconds": time.perf_counter() - start,
                  "model_loading_seconds": loading_seconds, "probe_seconds": probe_seconds,
                  "encoding_seconds": encoding_seconds, "index_save_seconds": time.perf_counter() - save_started,
                  "pages_per_second": len(pages) / encoding_seconds,
                  "command_seconds": time.perf_counter() - command_started,
                  "command_timing_scope": "main entry through index save; interpreter startup and final result printing excluded",
                  "memory_after_loading": memory_after_loading, "final_memory": process_memory(),
                  "index_bytes": sum(p.stat().st_size for p in Path(args.output).iterdir() if p.is_file()),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if config["device"] == "cuda" else None,
                  "timing_scope": "image reads, encoding and index save; excludes model loading and preprocessing probe",
                  "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None}
        write_json(Path(args.output) / "build.json", result)
        print(json.dumps(result, indent=2))
    else:
        from .evaluation import evaluate
        data = Path(args.data)
        if args.benchmark or args.query_config:
            from .query import benchmark
            result = benchmark(model, query_config, index, pages, read_rows(data / "queries.jsonl"), read_json(data / "qrels.json"),
                warmups=5 if args.benchmark else 1, repeats=3 if args.benchmark else 1)
            result["model_loading_seconds"] = loading_seconds
            result["index_bytes"] = sum(p.stat().st_size for p in Path(args.index).iterdir() if p.is_file())
        else:
            result = evaluate(model, config, index, pages, read_rows(data / "queries.jsonl"), read_json(data / "qrels.json"))
        if (data / "source.json").exists():
            result["dataset"] = read_json(data / "source.json")
            result["scope"] = result["dataset"].get("scope", result["scope"])
        write_json(Path(args.output) / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"results", "requests", "dataset"}}, indent=2))
    return 0


def entrypoint():
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"FolioRecall: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    entrypoint()
