"""Continue a published query student using cached teacher query vectors.

Uses Sentence Transformers' Trainer/checkpoints; only the pointwise objective and
the independent page-retrieval evaluation callback are project adaptations.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import read_json, write_json, write_rows, read_rows, provenance
from .query import load_query_encoder, encode_query_texts, validate_query_config
from .targets import load_targets


class QueryAlignmentLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels):
        student = self.model(sentence_features[0])["sentence_embedding"].float()
        teacher = labels.detach().float()
        if student.shape != teacher.shape or not torch.isfinite(teacher).all():
            raise ValueError("学生表示与教师目标形状或数值不匹配")
        loss = (1-F.cosine_similarity(student, teacher, dim=-1)).mean()
        if not torch.isfinite(loss):
            raise ValueError("蒸馏损失非有限值")
        return loss

    def get_config_dict(self):
        return {"objective": "mean(1-cosine(student, cached_teacher_query))"}


def distill(query_config, targets, output, index, data, limit=None, max_steps=None,
            micro_batch=32, max_seconds=3600, resume=None, evaluate_checkpoints=True):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from transformers import TrainerCallback, enable_full_determinism

    started = time.monotonic()
    if micro_batch not in (8, 16, 32):
        raise ValueError("微批应为8、16或32；有效batch保持32")
    if query_config["device"] != "cuda":
        raise ValueError("正式学生蒸馏配置必须指定 cuda；CPU 单元测试另行运行")
    rows, vectors, manifest = load_targets(targets, query_config, limit)
    teacher_config = manifest["encoding_config"]
    validate_query_config(query_config, teacher_config)
    from .search import load_index
    from .evaluation import validate_candidate_corpus
    dev_index, index_pages = load_index(index, teacher_config)
    validate_candidate_corpus(index_pages, read_rows(Path(data) / "pages.jsonl"))
    del dev_index, index_pages
    output = Path(output).resolve()
    settings = {"query_config": query_config, "targets": str(Path(targets).resolve()),
        "target_manifest": manifest, "count": len(rows), "epochs": 3, "learning_rate": 1e-5,
        "weight_decay": 0.01, "warmup_ratio": 0.03, "seed": 42, "effective_batch": 32,
        "micro_batch": micro_batch, "max_steps": max_steps, "data": str(Path(data).resolve()),
        "index": str(Path(index).resolve()), "evaluate_checkpoints": evaluate_checkpoints}
    if resume:
        resume = Path(resume).resolve()
        if resume.parent != output or read_json(output / "settings.json") != settings:
            raise ValueError("恢复必须使用所属输出目录、相同配置和目标")
        if read_rows(output / "training-queries.jsonl") != rows:
            raise ValueError("恢复查询 ID、文本或顺序变化")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("训练输出目录非空；恢复请指定 --resume")
        write_json(output / "settings.json", settings)
        write_rows(output / "training-queries.jsonl", rows)
        write_json(output / "teacher-config.json", teacher_config)
    run = output / "runs" / str(time.time_ns())
    provenance(run, dict(settings, resume=str(resume) if resume else None, max_seconds=max_seconds))
    enable_full_determinism(42)
    torch.set_num_threads(4)
    # Train FP32 master weights with BF16 autocast; exports retain FP32 weights.
    train_config = dict(query_config, dtype="float32")
    if resume:
        train_config["model_path"] = str(resume)
    model = load_query_encoder(train_config)
    loading_seconds = time.monotonic()-started
    selected = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
        if name.endswith("attention.q_lin.weight") or name in ("2.linear.weight", "3.linear.weight")}
    if not selected or not all(p.requires_grad for p in model.parameters()):
        raise ValueError("学生骨干或投影未加入训练")
    dataset = Dataset.from_dict({"query": [r["query"] for r in rows], "label": vectors.tolist()})
    history_path = output / "checkpoint-evaluations.json"
    evaluations = read_json(history_path) if history_path.exists() else []
    gradient_checks = []
    effective_limit = min(max_seconds, float(os.environ.get("FOLIORECALL_TASK_SECONDS", max_seconds))-30)

    class Checks(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if not gradient_checks:
                gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
                if any(g is None or not torch.isfinite(g).all() for g in gradients.values()):
                    raise ValueError("学生存在缺失或非有限梯度")
                nonzero = {name for name, g in gradients.items() if torch.count_nonzero(g).item() > 0}
                if not any("attention.q_lin" in n for n in nonzero) or not all(n in nonzero for n in ("2.linear.weight", "3.linear.weight")):
                    raise ValueError("骨干或投影没有有效梯度")
                gradient_checks.append({"finite_gradients": len(gradients), "nonzero_gradients": len(nonzero)})

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic()-started >= effective_limit:
                control.should_training_stop = True
                control.should_save = True
            if state.global_step % 10 == 0:
                print(f"distill step {state.global_step}/{state.max_steps}; {time.monotonic()-started:.1f}s", flush=True)

        def on_save(self, args, state, control, **kwargs):
            folder = output / f"checkpoint-{state.global_step}"
            export_config = dict(query_config, model_path=str(folder), dtype="bfloat16", batch_size=1)
            write_json(folder / "query-config.json", export_config)
            write_json(folder / "training-source.json", {"settings": str(output / "settings.json"), "run": str(run)})
            remaining = effective_limit-(time.monotonic()-started)
            if evaluate_checkpoints and remaining > 120:
                # A fresh BF16 model matches the public baseline's inference path. This
                # is quality evaluation while training is paused, not a memory benchmark.
                torch.cuda.empty_cache()
                command = [sys.executable, "-u", "-m", "foliorecall", "evaluate",
                    "--config", str(output / "teacher-config.json"), "--query-config", str(folder / "query-config.json"),
                    "--index", str(index), "--data", str(data), "--output", str(folder / "dev-evaluation")]
                with (folder / "evaluation.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
                result = read_json(folder / "dev-evaluation" / "result.json")
                evaluations.append({"step": state.global_step, "epoch": state.epoch,
                    "checkpoint": str(folder), "metrics": result["metrics"]})
                write_json(history_path, evaluations)
            elif evaluate_checkpoints:
                write_json(folder / "evaluation-pending.json", {"reason": "compute budget reserved for save/closeout"})

    arguments = SentenceTransformerTrainingArguments(output_dir=str(output), num_train_epochs=3,
        max_steps=max_steps if max_steps is not None else -1, per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=32//micro_batch, learning_rate=1e-5, weight_decay=0.01,
        warmup_ratio=0.03, lr_scheduler_type="linear", max_grad_norm=1.0, bf16=True,
        seed=42, data_seed=42, full_determinism=True, save_strategy="epoch", logging_steps=1,
        report_to="none", disable_tqdm=True, dataloader_num_workers=0, dataloader_pin_memory=False,
        optim="adamw_torch", remove_unused_columns=False)
    trainer = SentenceTransformerTrainer(model=model, args=arguments, train_dataset=dataset,
        loss=QueryAlignmentLoss(model), callbacks=[Checks()])
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        final = output / "final"
        trainer.save_model(str(final))
        export_config = dict(query_config, model_path=str(final), dtype="float32", device="cpu", batch_size=1)
        write_json(final / "query-config.json", export_config)
        after = dict(model.named_parameters())
        updated = [name for name, value in selected.items() if not torch.equal(value, after[name].detach().cpu())]
        if not any("attention.q_lin" in name for name in updated) or not all(name in updated for name in ("2.linear.weight", "3.linear.weight")):
            raise ValueError("骨干或投影未实际更新")
        # A small CPU roundtrip is independent of BF16 inference rounding.
        model.to(device="cpu", dtype=torch.float32).eval()
        probe = [r["query"] for r in rows[:4]]
        expected = encode_query_texts(model, probe, export_config)
        restored = load_query_encoder(export_config)
        actual = encode_query_texts(restored, probe, export_config)
        delta = float(np.max(np.abs(expected-actual)))
        if delta > 1e-6:
            raise ValueError(f"学生保存重载不一致: {delta}")
        result = {"complete": trainer.state.global_step >= trainer.state.max_steps,
            "completed_steps": trainer.state.global_step, "target_steps": trainer.state.max_steps,
            "seconds": time.monotonic()-started, "loading_seconds": loading_seconds,
            "gradient_checks": gradient_checks, "checked_updated_tensors": updated,
            "reload_max_abs_diff": delta, "evaluations": evaluations, "log_history": trainer.state.log_history,
            "training_dtype": "FP32 master weights, BF16 autocast", "final": str(final),
            "selection": "public initialization remains a candidate; deploy choice pending user review"}
        write_json(output / "result.json", result)
        write_json(run / "result.json", result)
        return result
    except Exception as exc:
        write_json(run / "failure.json", {"type": type(exc).__name__, "message": str(exc),
            "step": trainer.state.global_step, "seconds": time.monotonic()-started})
        raise
