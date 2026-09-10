"""Small fixed VDR subset; source Parquet files are processed one at a time."""
from collections import Counter
import io
from pathlib import Path
import random
import shutil
import subprocess
import time
import unicodedata
import urllib.request

from .io import read_rows, read_json, write_rows, write_json, provenance

VDR = "llamaindex/vdr-multilingual-train"
REVISION = "6b92b5cae23d44509f1e05d7062befe5ec77f7c9"


def normalized_query(value):
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def select_records(metadata, train_count=3000, dev_count=200, candidates=1000, seed=42):
    mapping = {row["id"]: row for row in metadata}
    if len(mapping) != len(metadata):
        raise ValueError("重复页面 ID")
    ids = sorted(mapping)
    random.Random(seed).shuffle(ids)
    dev_pool = set(ids[:len(ids) // 5])
    counts = Counter(normalized_query(row.get("query")) for row in metadata if normalized_query(row.get("query")))
    excluded = {row["id"] for row in metadata if counts.get(normalized_query(row.get("query")), 0) > 1}
    eligible = {"train": [], "dev": []}
    for row in metadata:
        pid = row["id"]
        if pid in excluded or not normalized_query(row.get("query")):
            continue
        name = "dev" if pid in dev_pool else "train"
        valid, rejected = [], []
        for negative in row.get("negatives") or []:
            reason = ("missing" if negative not in mapping else "self" if negative == pid else
                      "duplicate_query_page" if negative in excluded else
                      "cross_split" if (negative in dev_pool) != (pid in dev_pool) else None)
            if reason:
                rejected.append({"page_id": negative, "reason": reason})
            elif negative not in valid:
                valid.append(negative)
        if valid:
            eligible[name].append({"query_id": pid, "query": row["query"], "positive": pid,
                                   "negative": valid[0], "valid_negative_ids": valid,
                                   "original_negative_ids": row["negatives"], "rejected_negatives": rejected})
    chosen = {}
    for name, count in (("train", train_count), ("dev", dev_count)):
        if len(eligible[name]) < count:
            raise ValueError(f"{name} 有效查询不足：{len(eligible[name])} < {count}")
        chosen[name] = random.Random(seed).sample(sorted(eligible[name], key=lambda x: x["query_id"]), count)
    train_ids = {row[key] for row in chosen["train"] for key in ("positive", "negative")}
    dev_ids = {row["positive"] for row in chosen["dev"]}
    dev_ids.update(n for row in chosen["dev"] for n in row["valid_negative_ids"])
    rest = sorted(dev_pool - excluded - dev_ids)
    random.Random(seed).shuffle(rest)
    dev_ids.update(rest[:max(0, candidates - len(dev_ids))])
    if train_ids & dev_ids:
        raise ValueError("训练开发页面交叉")
    return chosen, train_ids, dev_ids, {"excluded_duplicate_query_pages": sorted(excluded),
        "eligible_queries": {k: len(v) for k, v in eligible.items()},
        "page_split": "sorted IDs shuffled with seed; first 20% dev", "seed": seed}


def download_shard(url, target, size):
    """Resume this task's temporary file, never touch shared HF cache."""
    offset = target.stat().st_size if target.exists() else 0
    if offset > size:
        raise ValueError(f"临时分片尺寸异常：{target}")
    control = target.with_name(target.name + ".aria2")
    marker = target.with_name(target.name + ".complete.json")
    if offset == size and not control.exists() and marker.exists():
        if read_json(marker) != {"url": url, "bytes": size}:
            raise ValueError("下载完成标记与当前源不一致")
        return 0
    if shutil.which("aria2c"):
        if offset and not control.exists():
            # aria2 can preallocate an incomplete file to the full source size.
            # Without its piece map or our completion marker no prefix is trusted.
            target.unlink()
            offset = 0
        command = ["aria2c", "--continue=true", "--allow-overwrite=false", "--auto-file-renaming=false",
                        "--max-connection-per-server=16", "--split=16", "--min-split-size=8M",
                        "--auto-save-interval=5", "--max-tries=4", "--retry-wait=3", "--summary-interval=30", "--console-log-level=warn",
                        "--download-result=full", "--dir=" + str(target.parent), "--out=" + target.name, url]
        for attempt in range(4):
            if target.exists() and not control.exists():
                target.unlink()
            try:
                subprocess.run(command, check=True)
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
                print(f"retry connection: {target.name}, attempt {attempt + 2}/4", flush=True)
                time.sleep(2 ** attempt)
        if target.stat().st_size != size or control.exists():
            raise ValueError("aria2 分片下载未完整结束")
        write_json(marker, {"url": url, "bytes": size})
        return max(0, size - offset)
    if control.exists():
        raise ValueError("存在 aria2 分段续传文件，需要 aria2 完成恢复")
    if offset == size:
        target.unlink()
    transferred = 0
    for attempt in range(4):
        offset = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("服务器未接受续传范围，拒绝拼接")
                with target.open("ab" if offset else "wb") as handle:
                    last = time.monotonic()
                    while block := response.read(4 * 1024 * 1024):
                        handle.write(block)
                        transferred += len(block)
                        if time.monotonic() - last >= 30:
                            print(f"download {target.name}: {handle.tell()}/{size}", flush=True)
                            last = time.monotonic()
            if target.stat().st_size != size:
                raise OSError("下载未达到源分片尺寸")
            write_json(marker, {"url": url, "bytes": size})
            return transferred
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def prepare_stage2(output, metadata_path, manifest_only=False):
    from huggingface_hub import HfApi, hf_hub_url
    import pyarrow.parquet as pq
    from PIL import Image
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = read_rows(metadata_path)
    mapping = {r["id"]: r for r in metadata}
    split, train_ids, dev_ids, selection = select_records(metadata)
    selected_ids = train_ids | dev_ids
    selection.update({"dataset": VDR, "revision": REVISION, "train_queries": len(split["train"]),
        "dev_queries": len(split["dev"]), "train_pages": len(train_ids), "dev_pages": len(dev_ids),
        "selected_pages": len(selected_ids), "original_document_isolation": "unverified; page-level only",
        "external_test": "ViDoRe held out; unknown upstream overlaps not audited",
        "scope": "internal development; single-source task adaptation"})
    if (output / "split.json").exists() and read_json(output / "split.json") != split:
        raise ValueError("已有数据切分不匹配，请使用新目录")
    write_json(output / "split.json", split)
    write_json(output / "source.json", selection)
    run_dir = output / "preparation" / f"run-{time.time_ns()}"
    provenance(run_dir, selection)
    print({key: value for key, value in selection.items() if key != "excluded_duplicate_query_pages"}, flush=True)
    if manifest_only:
        return
    files = sorted((f for f in HfApi().list_repo_tree(VDR, path_in_repo="en", repo_type="dataset", revision=REVISION)
                    if f.path.endswith(".parquet")), key=lambda f: f.path)
    image_dir, temporary = output / "images", output / "temporary-shards"
    image_dir.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    journal_path = output / "extraction.json"
    journal = read_json(journal_path) if journal_path.exists() else {"shards": [], "pages": []}
    completed = {r["source_shard"] for r in journal["shards"]}
    row_base = 0
    for file in files:
        if file.path in completed:
            record = next(r for r in journal["shards"] if r["source_shard"] == file.path)
            if not all(Path(p["preview"]).is_file() for p in journal["pages"] if p["source_shard"] == file.path):
                raise ValueError("已完成分片的页面缺失，请定向修复")
            row_base += record["rows"]
            continue
        target = temporary / Path(file.path).name
        started = time.monotonic()
        print(f"fetch {file.path} {file.size} bytes", flush=True)
        transferred = download_shard(hf_hub_url(VDR, file.path, repo_type="dataset", revision=REVISION), target, file.size)
        parquet = pq.ParquetFile(target)
        shard_ids = parquet.read(columns=["id"]).column("id").to_pylist()
        wanted = {i for i, pid in enumerate(shard_ids) if pid in selected_ids}
        cursor = 0
        for batch in parquet.iter_batches(batch_size=32, columns=["id", "image"]):
            if not any(i in wanted for i in range(cursor, cursor + len(batch))):
                cursor += len(batch)
                continue
            for local, row in enumerate(batch.to_pylist()):
                index = cursor + local
                if index not in wanted:
                    continue
                pid = row["id"]
                if mapping[pid]["row_index"] != row_base + index:
                    raise ValueError(f"源 ID / 行号不匹配：{pid}")
                content = row["image"]["bytes"]
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    width, height, fmt = image.width, image.height, image.format
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt, ".img")
                preview = image_dir / (pid + suffix)
                preview.write_bytes(content)
                journal["pages"].append({"page_id": pid, "doc_id": pid, "source": f"https://huggingface.co/datasets/{VDR}/blob/{REVISION}/{file.path}",
                    "page_number": 1, "page_number_kind": "standalone_image_not_original_pdf",
                    "preview": str(preview), "row_index": row_base + index, "source_shard": file.path,
                    "shard_row_index": index, "width": width, "height": height, "bytes": len(content),
                    "has_query": bool(normalized_query(mapping[pid].get("query")))})
            cursor += len(batch)
        journal["shards"].append({"source_shard": file.path, "rows": len(shard_ids), "source_bytes": file.size,
            "preparation_run": str(run_dir),
            "new_payload_bytes_lower_bound": transferred, "traffic_note": "excludes retries and protocol overhead; resumed preallocated files may undercount",
            "selected_pages": len(wanted), "seconds": time.monotonic() - started})
        pending = journal_path.with_name("extraction.pending.json")
        write_json(pending, journal)
        pending.replace(journal_path)
        # This exact task-owned file was just extracted successfully.
        target.unlink()
        target.with_name(target.name + ".complete.json").unlink(missing_ok=True)
        row_base += len(shard_ids)
        print(f"extracted {file.path}: {len(wanted)} pages; total {len(journal['pages'])}", flush=True)
    pages = sorted(journal["pages"], key=lambda p: p["row_index"])
    if len(pages) != len(selected_ids) or {p["page_id"] for p in pages} != selected_ids:
        raise ValueError("提取页面集合不完整或重复")
    write_rows(output / "pages.jsonl", pages)
    dev = output / "dev"
    write_rows(dev / "pages.jsonl", [p for p in pages if p["page_id"] in dev_ids])
    write_rows(dev / "queries.jsonl", [{"query_id": r["query_id"], "query": r["query"]} for r in split["dev"]])
    write_json(dev / "qrels.json", {r["query_id"]: {r["positive"]: 1} for r in split["dev"]})
    write_json(dev / "source.json", selection)
    selection.update({"saved_image_bytes": sum(p["bytes"] for p in pages),
        "negative_only_pages": sum(not p["has_query"] for p in pages), "complete": True})
    write_json(output / "source.json", selection)
