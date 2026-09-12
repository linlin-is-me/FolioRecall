"""First integration check against each package's native encode interface."""
import json
from pathlib import Path

import numpy as np
import torch

from foliorecall.io import read_json, write_json, provenance
from foliorecall.query import load_query_encoder, encode_query_texts, validate_query_config

torch.set_num_threads(4)
texts = ["What was the revenue growth in Q3?", "Which figure compares population projections?", "a " * 600]
root = Path("outputs/stage3/integration")
if (root / "result.json").exists():
    raise ValueError("integration result exists")
results = {}
for name in ("en", "ml"):
    config = read_json(f"configs/student-{name}-cpu.json")
    validate_query_config(config, read_json("configs/baseline.json"))
    model = load_query_encoder(config)
    actual = encode_query_texts(model, texts, config)
    native = np.asarray(model.encode(texts, convert_to_numpy=True), dtype=np.float32)
    delta = float(np.max(np.abs(actual-native)))
    if delta > 1e-6:
        raise ValueError(f"native representation mismatch: {delta}")
    results[name] = {"native_max_abs_diff": delta, "finite": bool(np.isfinite(actual).all()),
        "norms": np.linalg.norm(actual, axis=1).tolist(), "shape": list(actual.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": config, "model_bytes": sum(p.numel()*p.element_size() for p in model.parameters())}
    del model
provenance(root, results)
write_json(root / "result.json", results)
print(json.dumps(results, indent=2))
