"""Download pinned public ST query packages into the existing HF cache."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

for name in ("en", "ml"):
    config = json.loads(Path(f"configs/student-{name}-cpu.json").read_text())
    path = snapshot_download(config["model_id"], revision=config["revision"],
        allow_patterns=["*.json", "*.txt", "*.safetensors", "2_Dense/*.bin", "3_Dense/*.bin", "README.md"], max_workers=4)
    print(name, path, flush=True)
