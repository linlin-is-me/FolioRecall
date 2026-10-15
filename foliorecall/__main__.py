import argparse
import json
from pathlib import Path
import sys
import time

from .io import read_json, read_rows, write_json, provenance


def main():
    command_started = time.monotonic()
    parser = argparse.ArgumentParser(description="FolioRecall visual document retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    text_index = sub.add_parser('index-text', help='建立按页面 ID 绑定的 BM25 文本索引')
    text_index.add_argument('--pages', required=True)
    text_index.add_argument('--texts', required=True)
    text_index.add_argument('--output', required=True)
    for name in ('ask','evaluate-qa'):
        qa = sub.add_parser(name, help='公开文档的有限问答；需要北京百炼 API')
        qa.add_argument('--index', required=True)
        qa.add_argument('--config', required=True)
        qa.add_argument('--query-config', required=True)
        qa.add_argument('--retrieval-config')
        qa.add_argument('--agent-config', required=True)
        qa.add_argument('--public-corpus', required=True, help='确认可发送的公开页面和来源清单')
        qa.add_argument('--mode', choices=('rag','agent','multi-agent') if name == 'ask' else ('rag','agent'), required=True)
        qa.add_argument('--output', required=True)
        if name == 'ask':
            qa.add_argument('question')
            qa.add_argument('--resume', action='store_true', help='仅恢复 multi-agent 的原任务目录')
        else:
            qa.add_argument('--tasks', required=True)
            qa.add_argument('--repeats', type=int, default=1)
            qa.add_argument('--probe', required=True, help='同一固定模型及地域的已通过能力探针 result.json')
            qa.add_argument('--resume', action='store_true')
            qa.add_argument('--stop-after', type=int)
            qa.add_argument('--max-seconds', type=float)
            qa.add_argument('--dry-run', action='store_true')
            qa.add_argument('--translation-cache')
    text_eval = sub.add_parser("evaluate-text", help="上游页面文本的固定 BM25 对照")
    text_eval.add_argument("--config", required=True)
    text_eval.add_argument("--data", required=True)
    text_eval.add_argument("--output", required=True)
    serve = sub.add_parser("serve", help="启动单模型常驻的本地页面检索演示")
    serve.add_argument("--index", required=True)
    serve.add_argument("--config", required=True)
    serve.add_argument("--query-config", required=True)
    serve.add_argument('--retrieval-config')
    serve.add_argument('--agent-config')
    serve.add_argument('--multi-agent-config', help='可选的协作模式配置；需同时提供问答配置、公开语料和输出目录')
    serve.add_argument('--public-corpus')
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--output", help="可选的应用验证记录目录，不记录到已有实验中")
    imp = sub.add_parser("import")
    imp.add_argument("inputs", nargs="*")
    imp.add_argument("--image-manifest")
    imp.add_argument("--output", required=True)
    imp.add_argument("--dpi", type=int, default=150)
    imp.add_argument('--extract-text', action='store_true', help='提取 PDF 文本层；扫描页留空，不运行 OCR')
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
    student_train.add_argument("--depths", type=int, nargs='+', default=[6], help="Distillation exits: 6 (default), 4, or 4 6")
    student_train.add_argument("--max-seconds", type=float, default=3600)
    student_train.add_argument("--resume")
    student_train.add_argument("--stop-after", type=int, help="Pause at this global step without changing the training schedule")
    student_train.add_argument("--skip-evaluation", action="store_true", help="Save checkpoints without intermediate development evaluation")
    for name in ("index", "query", "evaluate", "train-smoke", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/lora-baseline.json" if name == "train" else "configs/baseline.json")
        if name == "index":
            command.add_argument("--pages", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--chunk-size", type=int, default=64)
            command.add_argument("--stop-after-chunks", type=int, help="Pause after this many newly saved chunks")
            command.add_argument("--max-seconds", type=float)
        elif name in ("query", "evaluate"):
            command.add_argument("--index", required=True)
            command.add_argument("--query-config")
            command.add_argument('--retrieval-config')
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
    from .retrieval import read_retrieval_config
    retrieval_config = read_retrieval_config(args.retrieval_config) if getattr(args, 'retrieval_config', None) else None
    if args.command == 'index-text':
        from .retrieval import build_text_index
        print(json.dumps(build_text_index(args.pages, args.texts, args.output), ensure_ascii=False, indent=2))
        return 0
    if args.command in ('ask','evaluate-qa'):
        if args.command == 'evaluate-qa':
            from .qa_runtime import prepare_unit, execute_plan, provider_factory
            spec = {key:getattr(args,key) for key in ('index','config','query_config','retrieval_config',
                    'agent_config','public_corpus','mode','tasks','probe')}
            unit = prepare_unit(spec,args.repeats,translation_cache=args.translation_cache)
            result = execute_plan(args.output,unit['identity'],unit['jobs'],lambda:provider_factory([unit]),
                [unit['agent']],args.resume,args.stop_after,args.max_seconds,args.dry_run,command_started)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['stopped'] is None else 75
        from .generation import read_agent_config, Bailian
        from .agent import ask, load_public_corpus
        from .search import load_index
        from .query import load_search_session
        teacher, student = read_json(args.config), read_json(args.query_config)
        agent_config = read_agent_config(args.agent_config)
        if args.resume and args.mode != 'multi-agent':
            raise ValueError('ask --resume 仅适用于 multi-agent')
        if Path(args.output).exists() and not (args.mode == 'multi-agent' and args.resume):
            raise ValueError('问答输出目录已存在，请使用新目录')
        if args.mode == 'multi-agent':
            from .multi_agent import ask_multi, require_multiagent
            require_multiagent()
        _, corpus_pages = load_index(args.index, teacher)
        evidence_texts = load_public_corpus(args.public_corpus, corpus_pages)
        if args.mode == 'multi-agent' and args.resume and (Path(args.output)/'result.json').exists():
            from .qa_runtime import transient
            saved = read_json(Path(args.output)/'result.json')
            if saved['completed'] or not transient(saved['termination_reason']) or len(list(Path(args.output).glob('attempt-*.json'))) >= 2:
                result = ask_multi(args.question,None,student,None,corpus_pages,agent_config,None,args.output,
                                   retrieval=retrieval_config,evidence_texts=evidence_texts,resume=True)
                print(json.dumps(result,ensure_ascii=False,indent=2))
                return 0 if result['completed'] else 75
        provider = Bailian(agent_config)
        loading_started = time.perf_counter()
        model, index, pages, retrieval = load_search_session(args.index,teacher,student,retrieval_config=retrieval_config)
        loading_seconds = time.perf_counter()-loading_started
        if args.command == 'ask':
            if args.mode == 'multi-agent':
                result=ask_multi(args.question,model,student,index,pages,agent_config,provider,args.output,
                                 retrieval,evidence_texts,resume=args.resume)
            else:
                provenance(args.output, {'page_config':teacher,'query_config':student,'agent_config':agent_config,
                    'retrieval_config':retrieval_config,'public_corpus':args.public_corpus,'mode':args.mode})
                result=ask(args.question,args.mode,model,student,index,pages,agent_config,provider,retrieval,evidence_texts)
            result['loading_seconds']=loading_seconds
            write_json(Path(args.output)/'result.json',result)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['completed'] else 75
    if args.command == "evaluate-text":
        from .text_baseline import evaluate_text
        result = evaluate_text(read_json(args.config), args.data, args.output)
        print(json.dumps(result['metrics'], indent=2))
        return 0
    if args.command == "serve":
        from .demo import serve_demo
        serve_demo(args.index, read_json(args.config), read_json(args.query_config), args.port, args.output,
                   retrieval_config=retrieval_config, agent_config_path=args.agent_config, public_corpus=args.public_corpus,
                   multi_agent_config_path=args.multi_agent_config)
        return 0
    if args.command == "cache-teacher":
        from .targets import cache_teacher
        print(json.dumps(cache_teacher(args.teacher, args.data, args.output, args.count, args.stop_after, args.max_seconds), indent=2))
        return 0
    if args.command == "distill":
        from .distillation import distill
        result = distill(read_json(args.query_config), args.targets, args.output, args.index, args.data,
            args.limit, args.max_steps, args.micro_batch, args.max_seconds, args.resume, not args.skip_evaluation, args.stop_after, args.depths)
        print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))
        return 0
    if args.command == "import":
        from .documents import import_documents
        result = import_documents(args.inputs, args.output, args.dpi, args.image_manifest, args.extract_text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["pages"] else 1
    config = read_json(args.config)
    query_config = read_json(args.query_config) if getattr(args, "query_config", None) else config
    if args.command == "index":
        from .indexing import build_index
        result = build_index(config, args.pages, args.output, args.resume, args.chunk_size, args.max_seconds, args.stop_after_chunks)
        print(json.dumps(result, indent=2))
        return 0 if result['complete'] else 75
    if args.command == "query":
        from .query import load_search_session, retrieve
        model, index, pages, retrieval = load_search_session(args.index, config,
            query_config if args.query_config else None, retrieval_config=retrieval_config)
        result, payload, _ = retrieve(model, query_config, index, pages, args.text, args.top_k, retrieval=retrieval)
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
    from .search import load_index
    from .query import load_query_encoder, benchmark
    from .indexing import index_sizes
    from .evaluation import evaluate, validate_candidate_corpus
    if args.query_config:
        from .query import validate_query_config
        validate_query_config(query_config, config)
    index, pages = load_index(args.index, config)
    data = Path(args.data)
    validate_candidate_corpus(pages, read_rows(data / "pages.jsonl"))
    from .retrieval import load_retrieval
    retrieval = load_retrieval(retrieval_config, pages)
    from .benchmark_data import load_dataset_provenance
    dataset = load_dataset_provenance(data) if (data / "source.json").exists() else None
    if (Path(args.output) / "result.json").exists():
        raise ValueError("评测结果已存在，请使用新输出目录")
    provenance(args.output, {"page_config": config, "query_config": query_config} if args.query_config else config)
    if retrieval_config:
        write_json(Path(args.output) / 'retrieval-config.json', retrieval_config)
    if args.benchmark or args.query_config or retrieval:
        import torch
        import faiss
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        faiss.omp_set_num_threads(1)
    loading_started = time.perf_counter()
    model = load_query_encoder(query_config)
    loading_seconds = time.perf_counter() - loading_started
    if args.benchmark or args.query_config or retrieval:
        result = benchmark(model, query_config, index, pages, read_rows(data / "queries.jsonl"), read_json(data / "qrels.json"),
            warmups=5 if args.benchmark else 1, repeats=3 if args.benchmark else 1, retrieval=retrieval)
        result["model_loading_seconds"] = loading_seconds
        result.update(index_sizes(args.index))
        result["index_bytes"] = result["deployment_index_bytes"]
        result["index_size_scope"] = "index.faiss + pages.jsonl + config.json; excludes cache/previews/run records"
        if retrieval:
            result['retrieval_config'] = retrieval_config
            result['text_index_bytes'] = retrieval['index_bytes']
            result['index_bytes'] += retrieval['index_bytes']
            result['index_size_scope'] += '; hybrid includes text index.json'
            result['timing_scope'] += '; includes BM25 Top-50 and RRF'
    else:
        result = evaluate(model, config, index, pages, read_rows(data / "queries.jsonl"), read_json(data / "qrels.json"))
    if dataset is not None:
        result["dataset"] = dataset
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
