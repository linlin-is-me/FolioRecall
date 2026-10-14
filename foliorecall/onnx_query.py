"""Native ORT backbone with the unchanged Sentence Transformers FP32 heads.

ST 6 needs Transformers 5, while published Optimum ONNX requires Transformers <4.58.
This small inference adapter avoids changing the existing runtime's core versions.
"""
from pathlib import Path
import numpy as np


class OnnxQueryEncoder:
    def __init__(self, config):
        import torch
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from sentence_transformers.models import Pooling, Dense, Normalize
        from .io import read_json
        root = Path(config['model_path'])
        backbone = read_json(root/'config.json')
        if tuple(backbone.get(k) for k in ('model_type','n_layers','dim','vocab_size')) != ('distilbert',6,768,30522):
            raise ValueError('ONNX学生骨干配置不匹配')
        modules = read_json(root/'modules.json')
        if [m['type'].split('.')[-1] for m in modules] != ['Transformer','Pooling','Dense','Dense','Normalize']:
            raise ValueError('ONNX学生缺少原池化、投影或归一化模块')
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.max_seq_length = config['max_length']
        classes = [Pooling, Dense, Dense, Normalize]
        self.heads = [cls.load(str(root/m['path'])) for cls,m in zip(classes,modules[1:])]
        if self.heads[0].pooling_mode != 'mean' or [
            (h.linear.in_features,h.linear.out_features) for h in self.heads[1:3]] != [(768,768),(768,2048)]:
            raise ValueError('ONNX学生池化或投影配置不匹配')
        if [type(h.activation_function).__name__ for h in self.heads[1:3]] != ['GELU','Identity']:
            raise ValueError('ONNX学生投影激活不匹配')
        for h in self.heads: h.float().eval()
        session = ort.SessionOptions()
        session.intra_op_num_threads = 4
        session.inter_op_num_threads = 1
        session.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(root/config['onnx_file']),session,
                                           providers=['CPUExecutionProvider'])
        self.input_names = [x.name for x in self.session.get_inputs()]

    def encode(self, texts, batch_size=1, **kwargs):
        import torch
        vectors = []
        with torch.inference_mode():
            for offset in range(0,len(texts),batch_size):
                tokens = self.tokenizer(texts[offset:offset+batch_size],padding=True,truncation=True,
                                        max_length=self.max_seq_length,return_tensors='np')
                feed = {name:np.asarray(tokens[name],dtype=np.int64) for name in self.input_names}
                hidden = self.session.run(None,feed)[0]
                features = {'token_embeddings':torch.from_numpy(hidden),
                            'attention_mask':torch.from_numpy(tokens['attention_mask'])}
                for layer in self.heads: features=layer(features)
                vectors.append(features['sentence_embedding'].numpy())
        return np.concatenate(vectors,axis=0)
