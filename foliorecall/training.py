import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from peft import LoraConfig
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.util import batch_to_device

from .encoding import load_encoder, encode_pages, encode_queries, processing
from .io import read_json, read_rows, write_json, provenance
from .search import save_index, load_index, search


def check_split(split, pages):
    page_ids = {p["page_id"] for p in pages}
    groups = []
    for name in ("train", "dev"):
        ids = []
        for row in split[name]:
            if not row["query"].strip() or row["positive"] == row["negative"]:
                raise ValueError("空查询或正负页面冲突")
            ids.extend((row["positive"], row["negative"]))
        if len(ids) != len(set(ids)) or not set(ids).issubset(page_ids):
            raise ValueError("重复或缺失页面；须重新组批")
        groups.append(set(ids))
    if groups[0] & groups[1]:
        raise ValueError("训练和开发页面交叉")


def train_smoke(config, data, output, checkpointing=False):
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise ValueError("当前 train-smoke 需要可用 CUDA；PDF 导入和 CPU 核心检查可继续")
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "adapter").exists():
        raise ValueError("产物目录已有适配器，请使用新目录，避免覆盖索引对应权重")
    torch.manual_seed(config["seed"])
    pages, split = read_rows(data / "pages.jsonl"), read_json(data / "split.json")
    check_split(split, pages)
    write_json(output / "data.json", {"source": read_json(data / "source.json"), "split": split, "pages": pages})
    mapping = {p["page_id"]: p for p in pages}
    provenance(output, dict(config, gradient_checkpointing=checkpointing))
    settings = config["train"]
    model = load_encoder(config)
    backbone = model[0].model
    targets = [name for name, _ in backbone.named_modules()
               if "language_model." in name and ".self_attn." in name
               and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    if not targets:
        raise ValueError("未找到语言侧注意力投影，拒绝扩大 LoRA 匹配范围")
    backbone.add_adapter(LoraConfig(r=settings["rank"], lora_alpha=settings["alpha"],
                                   lora_dropout=0.0, target_modules=targets, bias="none"))
    if checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n or "language_model." not in n for n in trainable):
        raise ValueError("可训练参数越过语言侧 LoRA 范围")
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=settings["learning_rate"])
    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=settings["mini_batch_size"])
    model.train()
    losses, steps = [], settings["steps"]
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        start = (step * settings["batch_size"]) % len(split["train"])
        batch = split["train"][start:start + settings["batch_size"]]
        groups = [[row["query"] for row in batch]]
        images = []
        for key in ("positive", "negative"):
            group = []
            for row in batch:
                with Image.open(mapping[row[key]]["preview"]) as image:
                    group.append(image.convert("RGB"))
            groups.append(group)
            images.extend(group)
        features = [batch_to_device(model.preprocess(group, prompt=config["prompt"],
                     processing_kwargs=processing(config)), model.device) for group in groups]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(features, None)
        if not torch.isfinite(loss):
            raise ValueError("训练损失非有限值")
        loss.backward()
        grads = [p.grad for p in trainable.values() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
            raise ValueError("LoRA 梯度缺失、非有限或全零")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("冻结参数出现梯度")
        optimizer.step()
        losses.append(float(loss.detach()))
        write_json(output / "progress.json", {"completed_steps": step + 1, "losses": losses})
        print(f"step {step + 1}/{steps} loss={losses[-1]:.6f}", flush=True)
        for image in images:
            image.close()
    updated = [name for name, p in trainable.items() if not torch.equal(initial[name], p.detach().cpu())]
    if not updated:
        raise ValueError("LoRA 参数未更新")
    elapsed, peak = time.perf_counter() - start_time, torch.cuda.max_memory_allocated()
    adapter = output / "adapter"
    backbone.save_pretrained(adapter)
    model.eval()
    probes = [mapping[row["positive"]] for row in split["dev"][:2]]
    before = encode_pages(model, probes, config)
    del optimizer, loss_fn, loss, features, trainable, backbone, model, grads
    gc.collect()
    torch.cuda.empty_cache()
    adapted = dict(config, adapter=str(adapter.resolve()))
    write_json(output / "config.json", adapted)
    model = load_encoder(adapted)
    after = encode_pages(model, probes, adapted)
    if not np.allclose(before, after, atol=1e-3, rtol=1e-3):
        raise ValueError("适配器保存重载后的向量不一致")
    dev_ids = {row[key] for row in split["dev"] for key in ("positive", "negative")}
    dev_pages = [row for row in pages if row["page_id"] in dev_ids]
    vectors = encode_pages(model, dev_pages, adapted)
    save_index(vectors, dev_pages, adapted, output / "index")
    index, restored = load_index(output / "index", adapted)
    queries = encode_queries(model, [row["query"] for row in split["dev"]], adapted)
    report = {"scope": "training/backward/reload flow check only", "gradient_checkpointing": checkpointing,
              "losses": losses, "updated_tensors": len(updated), "trainable_parameters": sum(x.numel() for x in initial.values()),
              "target_modules": targets, "frozen_gradients_absent": True, "reload_max_abs_diff": float(np.abs(before - after).max()),
              "training_seconds": elapsed, "peak_cuda_bytes": peak, "retrieval": search(index, restored, queries)}
    write_json(output / "result.json", report)
    return report
