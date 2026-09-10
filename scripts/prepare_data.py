"""Download only the stage-one inputs, retaining upstream IDs and revisions."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json, read_rows, write_json, write_rows

HR = "vidore/vidore_v3_hr"
HR_REV = "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3"
VDR = "llamaindex/vdr-multilingual-train"
VDR_REV = "6b92b5cae23d44509f1e05d7062befe5ec77f7c9"
PDFS = ["a_demographic_perspective_on_the_future_of_european-KJ0125152ENN.pdf",
        "employment_and_social_developments_in_europe-KE0125067ENN.pdf"]


def read_url(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except OSError as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch(repo, revision, filename):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def prepare_pdfs(output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = read_json(fetch(HR, HR_REV, "pdfs/metadata_european-reports.json"))
    sources = []
    for filename in PDFS:
        cached = fetch(HR, HR_REV, f"pdfs/{filename}")
        target = output / filename
        if not target.exists():
            target.symlink_to(cached)
        sources.append(dict(next(row for row in metadata if row["doc_name"] == filename),
                            local_path=str(target.absolute()), revision=HR_REV,
                            license_url=f"https://huggingface.co/datasets/{HR}"))
    write_json(output / "sources.json", sources)
    print(json.dumps(sources, indent=2), flush=True)


def prepare_hr(output):
    import pyarrow.parquet as pq
    import io
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    corpus = fetch(HR, HR_REV, "corpus/test-00000-of-00001.parquet")
    pages = []
    for batch in pq.ParquetFile(corpus).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            corpus_id = str(row["corpus_id"])
            target = image_dir / f"{corpus_id}.png"
            if not target.exists():
                with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
                    image.convert("RGB").save(target)
            doc_id = row["doc_id"]
            filename = doc_id if doc_id.endswith(".pdf") else doc_id + ".pdf"
            pages.append({"page_id": corpus_id, "row_index": len(pages), "doc_id": doc_id,
                          "page_number": int(row["page_number_in_doc"]) + 1,
                          "dataset_page_number": row["page_number_in_doc"], "preview": str(target.resolve()),
                          "source": f"https://huggingface.co/datasets/{HR}/resolve/{HR_REV}/pdfs/{filename}",
                          "document_name": filename})
    queries = pq.read_table(fetch(HR, HR_REV, "queries/test-00000-of-00001.parquet")).to_pylist()
    english = [row for row in queries if row["language"] in ("en", "english", "English")]
    english.sort(key=lambda row: str(row["query_id"]))
    selected = random.Random(42).sample(english, 20)
    selected = [{"query_id": str(q["query_id"]), "query": q["query"], "language": q["language"]} for q in selected]
    ids = {row["query_id"] for row in selected}
    qrels = {qid: {} for qid in ids}
    for row in pq.read_table(fetch(HR, HR_REV, "qrels/test-00000-of-00001.parquet")).to_pylist():
        qid = str(row["query_id"])
        if qid in ids:
            qrels[qid][str(row["corpus_id"])] = row["score"]
    assert len(pages) == 1110 and all(qrels.values()), (len(pages), len(english))
    write_rows(output / "pages.jsonl", pages)
    write_rows(output / "queries.jsonl", selected)
    write_json(output / "qrels.json", qrels)
    write_json(output / "source.json", {"dataset": HR, "revision": HR_REV, "seed": 42,
               "english_queries_total": len(english), "selected_queries": 20, "corpus_size": len(pages)})
    print(f"HR: {len(pages)} pages, {len(selected)} English queries", flush=True)


def prepare_vdr(output):
    from datasets import load_dataset
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "english-metadata.jsonl"
    if metadata_path.exists():
        metadata = read_rows(metadata_path)
    else:
        # Column projection avoids downloading the 20 GB image column.
        dataset = load_dataset(VDR, "en", split="train", revision=VDR_REV, streaming=True,
                               columns=["id", "query", "negatives", "language"])
        metadata = []
        for i, row in enumerate(dataset):
            metadata.append(dict(row, row_index=i))
            if i % 10000 == 0:
                print(f"VDR metadata {i}", flush=True)
        write_rows(metadata_path, metadata)
    mapping = {row["id"]: row for row in metadata}
    if len(mapping) != len(metadata):
        raise ValueError("VDR 页面 ID 不唯一")
    # No upstream-guaranteed original-document key: disclose page-level split.
    ids = sorted(mapping)
    random.Random(42).shuffle(ids)
    dev_ids = set(ids[:len(ids) // 5])
    chosen = {"train": [], "dev": []}
    used = set()
    # The viewer serves only a prefix of this large dataset. This tiny prefix is
    # sufficient for wiring checks; formal training needs source Parquet images.
    preview_limit = 1000
    for row in metadata:
        if row["row_index"] >= preview_limit or not (row.get("query") or "").strip():
            continue
        split = "dev" if row["id"] in dev_ids else "train"
        limit = 4 if split == "dev" else 8
        if len(chosen[split]) >= limit or row["id"] in used:
            continue
        negatives = [n for n in (row["negatives"] or []) if n in mapping and n != row["id"]
                     and mapping[n]["row_index"] < preview_limit
                     and (n in dev_ids) == (split == "dev") and n not in used]
        # Prefer nearby provided negatives to keep row-slice retrieval small.
        negatives.sort(key=lambda n: mapping[n]["row_index"])
        if not negatives:
            continue
        negative = negatives[0]
        original = row["negatives"] or []
        chosen[split].append({"query": row["query"], "positive": row["id"], "negative": negative,
                              "original_negative_ids": original,
                              "missing_negative_ids": [n for n in original if n not in mapping],
                              "excluded_cross_split_ids": [n for n in original if n in mapping
                                                            and (n in dev_ids) != (split == "dev")]})
        used.update((row["id"], negative))
        if len(chosen["train"]) == 8 and len(chosen["dev"]) == 4:
            break
    if len(chosen["train"]) != 8 or len(chosen["dev"]) != 4:
        raise ValueError("无法获得隔离且负例完整的最小样本")
    pages = []
    for pid in sorted(used):
        target = output / "images" / f"{pid}.jpg"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            params = urllib.parse.urlencode({"dataset": VDR, "config": "en", "split": "train",
                                            "offset": mapping[pid]["row_index"], "length": 1})
            data = json.loads(read_url("https://datasets-server.huggingface.co/rows?" + params))
            if not data.get("rows"):
                raise ValueError(f"切片服务没有返回页面 {pid}；原始页面未判定缺失")
            remote = data["rows"][0]["row"]
            if remote["id"] != pid or VDR_REV not in remote["image"]["src"]:
                raise ValueError("切片服务返回的 ID 或 revision 不匹配")
            content = read_url(remote["image"]["src"])
            import io
            from PIL import Image
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            target.write_bytes(content)
        pages.append({"page_id": pid, "doc_id": pid, "source": f"https://huggingface.co/datasets/{VDR}",
                      "page_number": 1, "page_number_kind": "standalone_image_not_original_pdf",
                      "preview": str(target.resolve())})
        print(f"VDR image {len(pages)}/{len(used)}", flush=True)
    write_rows(output / "pages.jsonl", pages)
    write_json(output / "split.json", chosen)
    write_json(output / "source.json", {"dataset": VDR, "revision": VDR_REV, "seed": 42,
               "split": "page ID; original-document isolation unverified", "metadata_pages": len(mapping),
               "downloaded_pages": len(pages), "queries": 12, "original_negatives_per_query": 1,
               "selection": "eligible records from first 1000 viewer rows, smoke test only; not representative"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["pdfs", "hr", "vdr", "model", "vdr-stage2"])
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.kind == "vdr-stage2":
        from foliorecall.data_preparation import prepare_stage2
        prepare_stage2("data/vdr-stage2", "data/vdr/english-metadata.jsonl", args.manifest_only)
    elif args.kind == "model":
        from huggingface_hub import snapshot_download
        config = read_json("configs/baseline.json")
        print(snapshot_download(config["model_id"], revision=config["revision"],
                                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"], max_workers=2))
    else:
        {"pdfs": prepare_pdfs, "hr": prepare_hr, "vdr": prepare_vdr}[args.kind](Path("data") / args.kind)
