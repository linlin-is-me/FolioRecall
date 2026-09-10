import argparse
import json
from pathlib import Path
import sys
import time

from .io import read_json, read_rows, write_json, provenance


def main():
    parser = argparse.ArgumentParser(description="FolioRecall visual document retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("inputs", nargs="*")
    imp.add_argument("--image-manifest")
    imp.add_argument("--output", required=True)
    imp.add_argument("--dpi", type=int, default=150)
    for name in ("index", "query", "evaluate", "train-smoke", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/lora-baseline.json" if name == "train" else "configs/baseline.json")
        if name == "index":
            command.add_argument("--pages", required=True)
            command.add_argument("--output", required=True)
        elif name in ("query", "evaluate"):
            command.add_argument("--index", required=True)
            if name == "query":
                command.add_argument("text")
                command.add_argument("--top-k", type=int, default=5)
                command.add_argument("--json", action="store_true")
            else:
                command.add_argument("--data", default="data/hr")
                command.add_argument("--output", required=True)
        else:
            command.add_argument("--data", default="data/vdr-stage2" if name == "train" else "data/vdr")
            command.add_argument("--output", required=True)
            command.add_argument("--gradient-checkpointing", action="store_true")
            if name == "train":
                command.add_argument("--resume")
                command.add_argument("--profile", action="store_true")
                command.add_argument("--stop-after", type=int)
                command.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    if args.command == "import":
        from .documents import import_documents
        result = import_documents(args.inputs, args.output, args.dpi, args.image_manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["pages"] else 1
    config = read_json(args.config)
    if args.command == "train":
        from .trainer import train
        if args.gradient_checkpointing:
            config["train"]["gradient_checkpointing"] = True
        try:
            train(config, args.data, args.output, args.resume, args.profile, args.stop_after, args.max_seconds)
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
        provenance(args.output, config)
    from .encoding import load_encoder, encode_pages, encode_queries
    model = load_encoder(config)
    if args.command == "index":
        import torch
        from PIL import Image
        from .encoding import processing
        with Image.open(pages[0]["preview"]) as image:
            features = model.preprocess([image.convert("RGB")], prompt=config["prompt"],
                                        processing_kwargs=processing(config))
        probe = {"first_page_id": pages[0]["page_id"], "max_seq_length": model.max_seq_length,
                 "pooling": model[1].pooling_mode, "input_ids_shape": list(features["input_ids"].shape),
                 "image_grid_thw": features["image_grid_thw"].tolist()}
        write_json(Path(args.output) / "encoding-probe.json", probe)
        del features
        start = time.perf_counter()
        vectors = encode_pages(model, pages, config)
        save_index(vectors, pages, config, args.output)
        result = {"pages": len(pages), "seconds": time.perf_counter() - start,
                  "timing_scope": "image reads, encoding and index save; excludes model loading and preprocessing probe",
                  "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None}
        write_json(Path(args.output) / "build.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "query":
        result = search(index, pages, encode_queries(model, [args.text], config), args.top_k)[0]
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for rank, row in enumerate(result, 1):
                print(f"{rank}. {row.get('document_name', row['doc_id'])} | page {row['page_number']} | {row['score']:.6f}")
                print(f"   source: {row['source']}\n   preview: {row['preview']}")
    else:
        from .evaluation import evaluate
        data = Path(args.data)
        result = evaluate(model, config, index, pages, read_rows(data / "queries.jsonl"), read_json(data / "qrels.json"))
        if (data / "source.json").exists():
            result["dataset"] = read_json(data / "source.json")
            result["scope"] = result["dataset"].get("scope", result["scope"])
        write_json(Path(args.output) / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"FolioRecall: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
