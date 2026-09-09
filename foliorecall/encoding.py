from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_encoder(config):
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；PDF 导入仍可使用 CPU，模型步骤尚未验证")
    model = SentenceTransformer(
        config["model_id"], revision=config["revision"], device=config["device"],
        model_kwargs={"dtype": getattr(torch, config["dtype"]), "attn_implementation": config["attn_implementation"]},
    )
    if model.get_embedding_dimension() != config["dimension"]:
        raise ValueError("模型维度与配置不符")
    model.max_seq_length = config["max_length"]
    if model[1].pooling_mode != config["pooling"]:
        raise ValueError("上游 pooling 与配置不符")
    if config.get("adapter"):
        model[0].model.load_adapter(str(Path(config["adapter"]).resolve()))
    model.eval()
    return model


def processing(config):
    return {"text": {"max_length": config["max_length"], "truncation": True},
            "image": {"min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"]}}


def encode(model, inputs, config):
    vectors = model.encode(inputs, batch_size=config["batch_size"], prompt=config["prompt"],
                           normalize_embeddings=config["normalize"], convert_to_numpy=True,
                           processing_kwargs=processing(config), show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != config["dimension"] or not np.isfinite(vectors).all():
        raise ValueError("模型输出维度或数值异常")
    if config["normalize"]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("模型输出零向量")
        # BF16 normalization has rounding error; FAISS cosine uses float32 unit vectors.
        vectors = vectors / norms
    return vectors


def encode_pages(model, rows, config):
    vectors = []
    for start in range(0, len(rows), config["batch_size"]):
        images = []
        for row in rows[start:start + config["batch_size"]]:
            with Image.open(row["preview"]) as image:
                images.append(image.convert("RGB"))
        vectors.append(encode(model, images, config))
        for image in images:
            image.close()
        if start % 25 == 0:
            print(f"encoded {min(start + len(images), len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(vectors)


def encode_queries(model, queries, config):
    return encode(model, queries, config)
