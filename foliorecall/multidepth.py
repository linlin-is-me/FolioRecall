"""Fixed 4/6-layer exits over the published query student, without a new encoder stack.

Depth readout and physical pruning follow Tevatron-Elastic's deployment idea:
https://github.com/zhichaoxu-shufe/tevatron-elastic
The separate projection heads and cached-target cosine objective are our adaptation.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .io import write_json


def validate_depths(depths, config):
    depths = tuple(depths)
    if depths not in ((6,), (4,), (4, 6)) or any(type(d) is not int for d in depths):
        raise ValueError('训练出口必须为6、4或4 6')
    if config.get('num_layers', 6) != max(depths):
        raise ValueError('训练出口与实际骨干层数不匹配')
    return depths


def readout(hidden, mask, modules):
    features = {'token_embeddings': hidden, 'attention_mask': mask}
    for module in modules:
        features = module(features)
    return features['sentence_embedding'].float()


class MultiDepthAlignmentLoss(nn.Module):
    def __init__(self, model, resume=None):
        super().__init__()
        self.model = model
        self.head4 = nn.ModuleList(deepcopy([model[i] for i in range(1, 5)]))
        # Trainer zeros/clips model.parameters(), not arbitrary loss parameters.
        # Register below Transformer, preserving the five-module ST forward path.
        model[0].add_module('depth4_head', self.head4)
        if resume:
            from safetensors.torch import load_file
            self.head4.load_state_dict(load_file(str(Path(resume)/'depth-4-head.safetensors')))

    def embeddings(self, features):
        # One backbone forward. Hidden state zero is the embedding lookup.
        output = self.model[0].model(input_ids=features['input_ids'],
            attention_mask=features['attention_mask'], output_hidden_states=True, return_dict=True)
        return {4: readout(output.hidden_states[4], features['attention_mask'], self.head4),
                6: readout(output.hidden_states[6], features['attention_mask'],
                           [self.model[i] for i in range(1, 5)])}

    def forward(self, sentence_features, labels):
        targets = labels.detach().float()
        representations = self.embeddings(sentence_features[0])
        if not torch.isfinite(targets).all() or any(v.shape != targets.shape for v in representations.values()):
            raise ValueError('多深度学生与教师目标形状或数值不匹配')
        loss = torch.stack([(1-F.cosine_similarity(v, targets, dim=-1)).mean()
                            for v in representations.values()]).mean()
        if not torch.isfinite(loss):
            raise ValueError('多深度损失非有限值')
        return loss

    def save_head(self, folder):
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head4.state_dict().items()},
                  str(Path(folder)/'depth-4-head.safetensors'))

    def get_config_dict(self):
        return {'depths': [4, 6], 'weights': [0.5, 0.5],
                'objective': 'mean over exits of mean(1-cosine(student, cached_teacher_query))'}


def export_exit(model, depth, folder, config, head=None):
    """Materialize a standard ST package; unused blocks and the other head are absent."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('出口导出目录非空')
    exported = deepcopy(model).to(device='cpu', dtype=torch.float32).eval()
    if hasattr(exported[0], 'depth4_head'):
        delattr(exported[0], 'depth4_head')
    backbone = exported[0].model
    if depth not in (4, 6) or depth > backbone.config.n_layers:
        raise ValueError('导出层数超过实际骨干')
    del backbone.transformer.layer[depth:]
    backbone.config.n_layers = depth
    backbone.config.output_hidden_states = False
    if head is not None:
        for i, module in enumerate(head, 1):
            exported[i] = deepcopy(module).cpu().float().eval()
    exported.save_pretrained(str(folder), create_model_card=False)
    cfg = dict(config, num_layers=depth, model_path=str(folder.resolve()), device='cpu', dtype='float32', batch_size=1)
    write_json(folder/'query-config.json', cfg)
    return exported, cfg


from sentence_transformers import SentenceTransformerTrainer


class MultiDepthTrainer(SentenceTransformerTrainer):
    def _load_from_checkpoint(self, checkpoint_path):
        # ST saves the HF backbone and normal heads; the callback saves head four.
        # Restore normal weights without treating that auxiliary head as missing.
        transformer = self.model[0]
        head = transformer.depth4_head
        delattr(transformer, 'depth4_head')
        try:
            super()._load_from_checkpoint(checkpoint_path)
        finally:
            transformer.add_module('depth4_head', head)
