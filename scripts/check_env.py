"""Offline environment smoke check; run on first setup or relevant changes only."""

import importlib.metadata
import json
import os
import platform
import sys

import accelerate
import faiss
import numpy as np
import pypdfium2 as pdfium
import sentence_transformers
import torch
import torchvision
from PIL import Image
from transformers import AutoProcessor, Qwen3VLModel


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the independent foliorecall_env first.")

    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "environment": sys.prefix,
        "hf_home": os.environ.get("HF_HOME"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch", "torchvision", "sentence-transformers", "transformers",
                "accelerate", "numpy", "pillow", "pypdfium2", "faiss-cpu",
            )
        },
    }

    # Synthetic blank page only: validates the rendering library, not PDF import.
    with pdfium.PdfDocument.new() as document:
        page = document.new_page(32, 32)
        try:
            bitmap = page.render(scale=1)
            try:
                with bitmap.to_pil() as image:
                    assert image.size == (32, 32)
            finally:
                bitmap.close()
        finally:
            page.close()
    report["synthetic_pdf_render"] = "passed"

    vectors = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    scores, ids = index.search(vectors[:1], 1)
    assert ids[0, 0] == 0 and scores[0, 0] == 1.0
    report["synthetic_faiss_search"] = "passed"

    cpu = torch.ones((32, 32))
    assert torch.equal(cpu @ cpu, torch.full_like(cpu, 32))
    report["torch_cpu"] = "passed"
    report["cuda_runtime"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        gpu = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        result = gpu @ gpu
        torch.cuda.synchronize()
        assert torch.equal(result.cpu(), torch.full((32, 32), 32, dtype=torch.bfloat16))
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_bf16_matmul"] = "passed"
    else:
        report["cuda_bf16_matmul"] = "skipped; CPU work remains available"
    report["model_loading_and_training"] = "not tested; no weights downloaded"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
