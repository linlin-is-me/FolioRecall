"""Ordinary LoRA training using Sentence Transformers' scheduling and GradCache.

Only collating local images, stable batching and adapter-only resume are adapted.
"""
import gc
from functools import partial
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from datasets import Dataset
from peft import LoraConfig
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import NoDuplicatesBatchSampler
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from transformers import TrainerCallback, set_seed

from .data_preparation import normalized_query
from .encoding import load_encoder, processing, encode_pages, encode_queries
from .evaluation import evaluate, validate_candidate_corpus
from .io import read_rows, read_json, write_rows, write_json, provenance
from .search import save_index, search, load_index


def validate_training_data(split, pages, dev_pages):
    mapping = {p["page_id"]: p for p in pages}
    if len(mapping) != len(pages):
        raise ValueError("重复页面 ID")
    groups, queries = [], []
    for name in ("train", "dev"):
        ids, texts = set(), set()
        for row in split[name]:
            text = normalized_query(row["query"])
            if not text or text in texts or row["positive"] == row["negative"]:
                raise ValueError("重复/空查询或正负页面冲突")
            texts.add(text)
            ids.update((row["positive"], row["negative"]))
            if row["negative"] not in row["original_negative_ids"]:
                raise ValueError("负例不在上游列表")
        if not ids.issubset(mapping):
            raise ValueError("缺失训练页面引用")
        groups.append(ids)
        queries.append(texts)
    candidate_ids = {p["page_id"] for p in dev_pages}
    if groups[0] & candidate_ids or groups[0] & groups[1] or queries[0] & queries[1]:
        raise ValueError("训练开发页面或查询交叉")
    if not groups[1].issubset(candidate_ids):
        raise ValueError("开发候选库缺失配对页面")
    if any(not Path(p["preview"]).is_file() for p in pages):
        raise ValueError("页面图像不存在")
    return mapping


def choose_profile(rows, mapping, count, seed):
    """Spread the small resource probe over image area and aspect ratio."""
    if len(rows) < count:
        raise ValueError("资源短跑样本不足")
    ordered = sorted(rows, key=lambda r: (mapping[r["positive"]]["width"] * mapping[r["positive"]]["height"],
                                          mapping[r["positive"]]["width"] / mapping[r["positive"]]["height"], r["query_id"]))
    # Pick across the full size range; this is a resource sample, not a quality estimate.
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[int(i)] for i in indices]


def training_dataset(rows, mapping):
    return Dataset.from_list([{"query": r["query"],
        "positive": {"path": str(Path(mapping[r["positive"]]["preview"]).resolve())},
        "negative": {"path": str(Path(mapping[r["negative"]]["preview"]).resolve())}} for r in rows])


def batch_plan(dataset, batch_size, seed):
    batches = list(NoDuplicatesBatchSampler(dataset, batch_size=batch_size, drop_last=False,
                    generator=torch.Generator(), seed=seed))
    flattened = [i for batch in batches for i in batch]
    if sorted(flattened) != list(range(len(dataset))):
        raise ValueError("组批遗漏或重复训练记录")
    for batch in batches:
        values = [dataset[i][key]["path"] for i in batch for key in ("positive", "negative")]
        if len(values) != len(set(values)):
            raise ValueError("逻辑 batch 页面冲突")
    return batches


def fixed_batch_sampler(dataset, batch_size, seed=0, *, fixed_seed=None, **kwargs):
    # ST does not forward args.data_seed to this callable. Bind our config seed
    # explicitly; a top-level partial also survives training_args serialization.
    if fixed_seed is None:
        raise ValueError("组批必须显式绑定配置 fixed_seed")
    return batch_plan(dataset, batch_size, fixed_seed)


def restore_history(output, start_step, run_directory):
    """Keep abandoned attempts, but count only steps on the resumed path."""
    progress = output / "progress.json"
    history = read_json(progress)["steps"] if progress.exists() else []
    retained = [row for row in history if row["step"] <= start_step]
    if [row["step"] for row in retained] != list(range(1, start_step + 1)):
        raise ValueError("进度历史与恢复检查点不一致：存在缺失或重复步骤")
    for path in [progress, *output.glob("result*.json")]:
        if path.exists():
            shutil.copy2(path, run_directory / ("previous-" + path.name))
    # A completed report from an abandoned path must not describe this attempt.
    (output / "result.json").unlink(missing_ok=True)
    write_json(progress, {"steps": retained, "completed_steps": start_step})
    return retained


def preprocessing(model, config):
    def prepare(inputs, prompt=None, task=None, **kwargs):
        images = []
        try:
            if isinstance(inputs[0], dict):
                for item in inputs:
                    with Image.open(item["path"]) as image:
                        images.append(image.convert("RGB"))
                inputs = images
            return model.preprocess(inputs, prompt=prompt, task=task, processing_kwargs=processing(config))
        finally:
            for image in images:
                image.close()
    return prepare


def checkpoint_adapter(folder):
    locations = list(Path(folder).rglob("adapter_config.json"))
    if len(locations) != 1:
        raise ValueError("检查点必须包含且仅包含一个适配器")
    return locations[0].parent


class AdapterTrainer(SentenceTransformerTrainer):
    def _load_from_checkpoint(self, checkpoint_path):
        adapter = checkpoint_adapter(checkpoint_path)
        # Transformers 5.16.1 incorrectly forwards its local_files_only argument
        # into LoadStateDictConfig. This is an existing verified local directory.
        result = self.model[0].model.load_adapter(str(adapter.resolve()), adapter_name="default", hotswap=True,
                                                is_trainable=True)
        if result and any(getattr(result, key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
            raise ValueError(f"适配器恢复 key 不匹配：{result}")
        if not all(p.requires_grad for n, p in self.model.named_parameters() if "lora_" in n):
            raise ValueError("恢复后 LoRA 未保持可训练")


class RunChecks(TrainerCallback):
    def __init__(self, model, output, config, stop_after, max_seconds, initial, dev, profile, history=None):
        self.model, self.output, self.config = model, output, config
        self.stop_after, self.max_seconds, self.initial = stop_after, max_seconds, initial
        self.dev, self.profile = dev, profile
        self.history = list(history or [])
        self.started = time.monotonic()

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        self.started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        self.initial.clear()
        self.initial.update({n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad})
        trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if trainable_ids != optimizer_ids:
            raise ValueError("优化器参数与 LoRA 可训练参数不一致")
        write_json(self.output / f"resume-state-{state.global_step}.json", {
            "global_step": state.global_step, "optimizer_state_entries": len(optimizer.state),
            "optimizer_steps": sorted({int(v["step"]) for v in optimizer.state.values() if "step" in v}),
            "learning_rates": [g["lr"] for g in optimizer.param_groups]})

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.step_started = time.monotonic()
        self.step_lr = kwargs["optimizer"].param_groups[0]["lr"]

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
            raise ValueError("LoRA 梯度缺失、非有限或全零")
        if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad):
            raise ValueError("冻结参数出现梯度")

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.history.append({"step": state.global_step, "seconds": time.monotonic() - self.step_started,
                             "learning_rate_used": self.step_lr,
                             "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
        write_json(self.output / "progress.json", {"steps": self.history, "completed_steps": state.global_step})
        if (self.stop_after and state.global_step >= self.stop_after) or time.monotonic() - self.started >= self.max_seconds:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and not math.isfinite(logs["loss"]):
            raise ValueError("训练损失非有限值")

    def on_save(self, args, state, control, **kwargs):
        folder = self.output / f"checkpoint-{state.global_step}"
        config = dict(self.config, adapter=str(checkpoint_adapter(folder).resolve()))
        write_json(folder / "encoding.json", config)
        if not self.profile:
            pages = read_rows(self.dev / "pages.jsonl")
            was_training = self.model.training
            try:
                self.model.eval()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                vectors = encode_pages(self.model, pages, config)
                index = save_index(vectors, pages, config, folder / "dev-index")
                write_json(folder / "dev-build.json", {"pages": len(pages), "seconds": time.monotonic() - started,
                           "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                           "timing_scope": "page encoding and index save with training model resident"})
                result = evaluate(self.model, config, index, pages, read_rows(self.dev / "queries.jsonl"), read_json(self.dev / "qrels.json"))
                result["scope"] = "internal development; not external test"
                write_json(folder / "dev-result.json", result)
            finally:
                self.model.train(was_training)


def train(config, data, output, resume=None, profile=False, stop_after=None, max_seconds=3600):
    invocation_started = time.monotonic()
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise ValueError("普通 LoRA 训练需要可用 CUDA")
    if config.get("adapter"):
        raise ValueError("首轮训练从原始模型初始化；恢复请使用 --resume")
    data, output = Path(data).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    split, pages = read_json(data / "split.json"), read_rows(data / "pages.jsonl")
    mapping = validate_training_data(split, pages, read_rows(data / "dev/pages.jsonl"))
    settings = config["train"]
    if settings["epochs"] != 1 or settings["gradient_accumulation_steps"] != 1:
        raise ValueError("当前入口只实现单遍、梯度累积为 1 的对照训练")
    rows = choose_profile(split["train"], mapping, settings["profile_queries"], config["seed"]) if profile else split["train"]
    dataset = training_dataset(rows, mapping)
    batches = batch_plan(dataset, settings["batch_size"], config["seed"])
    max_steps = settings["profile_steps"] if profile else len(batches)
    if profile and (len(batches) != max_steps or any(len(b) != settings["batch_size"] for b in batches)):
        raise ValueError("资源样本存在批冲突，须重新选择 32 条配对后短跑")
    run_settings = {"config": config, "profile": profile, "data": str(data), "rows": rows, "batches": batches,
                    "max_steps": max_steps, "batching_version": 2, "batching_seed": config["seed"]}
    if (output / "settings.json").exists():
        if read_json(output / "settings.json").get("batching_version") != 2:
            raise ValueError("旧运行未绑定配置组批 seed，不能按新规则恢复；请保留旧产物并使用新输出目录")
        if not resume or read_json(output / "settings.json") != run_settings:
            raise ValueError("已有训练目录或恢复配置不一致，请使用新输出目录")
    write_json(output / "settings.json", run_settings)
    write_rows(output / "samples.jsonl", rows)
    start_step = int(read_json(Path(resume) / "trainer_state.json")["global_step"]) if resume else 0
    if resume and Path(resume).resolve().parent != output:
        raise ValueError("恢复检查点必须属于本训练输出目录")
    if start_step >= max_steps:
        raise ValueError("检查点已达到训练目标，无需恢复")
    run_directory = output / f"run-from-{start_step}-{time.time_ns()}"
    provenance(run_directory, dict(config, profile=profile, resume=resume,
                                                      stop_after=stop_after, max_seconds=max_seconds))
    history = restore_history(output, start_step, run_directory)
    set_seed(config["seed"])
    started = time.monotonic()
    model = load_encoder(config)
    model.model_card_data.generate_widget_examples = False
    model.model_card_data.local_files_only = True
    loading_seconds = time.monotonic() - started
    backbone = model[0].model
    targets = [n for n, _ in backbone.named_modules() if "language_model." in n and ".self_attn." in n
               and n.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    if not targets:
        raise ValueError("语言侧 LoRA 目标为空")
    backbone.add_adapter(LoraConfig(r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
                                   target_modules=targets, bias="none"))
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    if not initial or any("lora_" not in n or "language_model." not in n for n in initial):
        raise ValueError("可训练参数越过语言侧 LoRA")
    collator = SentenceTransformerDataCollator(preprocess_fn=preprocessing(model, config), prompts=config["prompt"])
    # Record and compare the new collator with the already-validated direct path.
    first = dataset[0]
    prepared = collator([first])
    with Image.open(first["positive"]["path"]) as image:
        direct = model.preprocess([image.convert("RGB")], prompt=config["prompt"], processing_kwargs=processing(config))
    matched = all(torch.equal(value, prepared["positive_" + key]) for key, value in direct.items() if torch.is_tensor(value))
    if not matched:
        raise ValueError("新 collator 与原图像预处理不一致")
    write_json(output / "preprocessing-check.json", {"matched": matched, "image_grid_thw": direct["image_grid_thw"].tolist(),
        "input_ids_shape": list(direct["input_ids"].shape), "target_modules": targets})
    del direct, prepared
    args = SentenceTransformerTrainingArguments(output_dir=str(output), per_device_train_batch_size=settings["batch_size"],
        num_train_epochs=1, max_steps=max_steps, learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
        lr_scheduler_type="linear", warmup_steps=math.ceil(max_steps * settings["warmup_ratio"]), optim="adamw_torch",
        gradient_accumulation_steps=1, max_grad_norm=settings["max_grad_norm"], bf16=True,
        gradient_checkpointing=settings["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, logging_nan_inf_filter=False, save_steps=max(1, max_steps // 2), save_strategy="steps",
        eval_strategy="no", load_best_model_at_end=False, seed=config["seed"], data_seed=config["seed"],
        report_to="none", disable_tqdm=True, dataloader_num_workers=0, dataloader_pin_memory=False,
        batch_sampler=partial(fixed_batch_sampler, fixed_seed=config["seed"]))
    checks = RunChecks(model, output, config, stop_after, max_seconds, initial, data / "dev", profile, history)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=settings["mini_batch_size"], scale=settings["scale"])
    trainer = AdapterTrainer(model=model, args=args, train_dataset=dataset, loss=loss, data_collator=collator, callbacks=[checks])
    trained = time.monotonic()
    trainer.train(resume_from_checkpoint=resume)
    updated = [n for n, p in model.named_parameters() if n in initial and not torch.equal(initial[n], p.detach().cpu())]
    if not updated and (trainer.state.global_step >= max_steps or any(
            step.get("learning_rate_used", 0) > 0 for step in checks.history if step["step"] > start_step)):
        raise ValueError("LoRA 参数没有更新")
    report = {"scope": "resource/backward/resume check only" if profile else "ordinary LoRA internal-development baseline",
        "completed_steps": trainer.state.global_step, "target_steps": max_steps,
        "complete": trainer.state.global_step >= max_steps, "model_loading_seconds": loading_seconds,
        "segment_seconds": time.monotonic() - trained,
        "peak_cuda_bytes": max((step["peak_cuda_bytes"] for step in checks.history), default=0),
        "peak_cuda_scope": "maximum training-step allocation on the effective resumed path; development build/query reported separately",
        "updated_tensors": len(updated), "trainable_parameters": sum(p.numel() for p in initial.values()),
        "parameter_update_check": "passed" if updated else "pending; only zero-learning-rate warmup executed",
        "frozen_gradients_absent": True, "steps": checks.history, "log_history": trainer.state.log_history}
    report["completed_query_presentations"] = sum(len(batches[step["step"] - 1]) for step in checks.history)
    report["step_training_seconds"] = sum(step["seconds"] for step in checks.history)
    report["step_timing_scope"] = "forward/backward/optimizer; excludes DataLoader collation and checkpoint I/O"
    report["segment_timing_scope"] = "Trainer.train wall time, including collation, checkpoint I/O and development evaluation; excludes initial model load and export reload"
    report["invocation_seconds"] = time.monotonic() - invocation_started
    write_json(output / f"result-step-{trainer.state.global_step}.json", report)
    if report["complete"]:
        adapter = checkpoint_adapter(output / f"checkpoint-{trainer.state.global_step}")
        adapted = dict(config, adapter=str(adapter.resolve()))
        write_json(output / "config.json", adapted)
        probes = [mapping[r["positive"]] for r in rows[:2]]
        model.eval()
        before = encode_pages(model, probes, adapted)
        del trainer, loss, checks, backbone, model, collator
        gc.collect()
        torch.cuda.empty_cache()
        model = load_encoder(adapted)
        after = encode_pages(model, probes, adapted)
        if not np.allclose(before, after, atol=1e-3, rtol=1e-3):
            raise ValueError("适配器重载向量不一致")
        index = save_index(after, probes, adapted, output / "probe-index")
        index, restored = load_index(output / "probe-index", adapted)
        report["reload_max_abs_diff"] = float(np.abs(before - after).max())
        report["probe_retrieval"] = search(index, restored, encode_queries(model, [rows[0]["query"]], adapted))
        report["invocation_seconds"] = time.monotonic() - invocation_started
        write_json(output / "result.json", report)
    return report
