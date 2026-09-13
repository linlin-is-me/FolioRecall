from pathlib import Path
import uuid

from PIL import Image

from .io import read_json, read_rows, write_json, write_rows


def validate_pages(rows):
    ids = [row["page_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("重复 page_id")
    for row in rows:
        for key in ("doc_id", "source", "preview", "page_number"):
            if key not in row:
                raise ValueError(f"页面缺少 {key}: {row['page_id']}")
        if not isinstance(row["page_number"], int) or row["page_number"] < 1:
            raise ValueError("page_number 必须是从 1 开始的物理页码")
        if not Path(row["preview"]).is_file():
            raise ValueError(f"预览不存在: {row['preview']}")
    return rows


def import_documents(inputs, output, dpi=150, image_manifest=None):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError("文档导入需要安装 foliorecall[image]") from exc
    if dpi <= 0:
        raise ValueError("DPI 必须为正")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Reject changes before writing any previews or replacing the old manifest.
    for item in inputs:
        source = Path(item).absolute()
        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, source.as_uri()).hex
        saved = output / doc_id / "import.json"
        if saved.exists():
            try:
                stat = source.stat()
            except OSError as exc:
                raise ValueError(f"已导入的源文件无法读取: {source}；请使用新输出目录") from exc
            signature = {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "dpi": dpi}
            if read_json(saved)["input"] != signature:
                raise ValueError(f"输入或 DPI 已变化: {source}；请使用新输出目录，避免改写旧索引预览")
    rows, errors = [], []
    if image_manifest:
        base = Path(image_manifest).resolve().parent
        for row in read_rows(image_manifest):
            row = dict(row)
            row["preview"] = str((base / row["preview"]).resolve())
            rows.append(row)
    for source in inputs:
        # Preserve the user-facing filename when HF cache inputs are symlinks.
        source = Path(source).absolute()
        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, source.as_uri()).hex
        folder = output / doc_id
        folder.mkdir(exist_ok=True)
        doc_rows = []
        try:
            stat = source.stat()
            signature = {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "dpi": dpi}
            saved = folder / "import.json"
            if saved.exists() and read_json(saved)["input"] == signature:
                cached = read_json(saved)["pages"]
                validate_pages(cached)
                rows.extend(cached)
                continue
            if source.suffix.lower() == ".pdf":
                with pdfium.PdfDocument(source) as doc:
                    for index in range(len(doc)):
                        preview = folder / f"{index + 1:04d}.png"
                        page = doc[index]
                        bitmap = page.render(scale=dpi / 72)
                        bitmap.to_pil().convert("RGB").save(preview)
                        bitmap.close()
                        page.close()
                        doc_rows.append({"page_id": f"{doc_id}:{index + 1}", "doc_id": doc_id,
                                         "source": str(source), "document_name": source.name,
                                         "page_number": index + 1, "preview": str(preview)})
            else:
                preview = folder / "0001.png"
                with Image.open(source) as image:
                    image.convert("RGB").save(preview)
                doc_rows.append({"page_id": f"{doc_id}:1", "doc_id": doc_id, "source": str(source),
                                 "document_name": source.name, "page_number": 1, "preview": str(preview)})
            write_json(saved, {"input": signature, "pages": doc_rows})
            rows.extend(doc_rows)
        except (OSError, ValueError, pdfium.PdfiumError) as exc:
            errors.append({"source": str(source), "error": str(exc)})
    validate_pages(rows)
    write_rows(output / "pages.jsonl", rows)
    write_json(output / "import-errors.json", errors)
    return {"pages": len(rows), "errors": errors, "manifest": str(output / "pages.jsonl")}
