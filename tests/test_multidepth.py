import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from foliorecall.multidepth import MultiDepthAlignmentLoss, MultiDepthTrainer, export_exit, validate_depths
from foliorecall.query import validate_backend


class MultiDepthTests(unittest.TestCase):
    def tiny(self, root):
        from transformers import DistilBertConfig, DistilBertModel, BertTokenizer
        from sentence_transformers import SentenceTransformer, models
        source = root/'source'
        source.mkdir()
        (source/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\none\ntwo\nthree\n')
        BertTokenizer(vocab_file=str(source/'vocab.txt')).save_pretrained(source)
        DistilBertModel(DistilBertConfig(vocab_size=8, dim=12, hidden_dim=24, n_layers=6,
            n_heads=3, dropout=0, attention_dropout=0)).save_pretrained(source)
        return SentenceTransformer(modules=[models.Transformer(str(source),max_seq_length=512),
            models.Pooling(12,pooling_mode='mean'), models.Dense(12,12,activation_function=torch.nn.GELU()),
            models.Dense(12,7,activation_function=torch.nn.Identity()), models.Normalize()],device='cpu')

    def test_gradients_mask_optimizer_head_resume_and_physical_export(self):
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            torch.manual_seed(42)
            model=self.tiny(root)
            loss=MultiDepthAlignmentLoss(model)
            features=model.tokenize(['one two','three'])
            targets=torch.randn(2,7,requires_grad=True)
            calls=[]
            hook=model[0].model.register_forward_hook(lambda *args:calls.append(1))
            value=loss([features],targets)
            hook.remove()
            self.assertEqual(len(calls),1)
            value.backward()
            self.assertIsNone(targets.grad)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in loss.parameters()))
            args=SentenceTransformerTrainingArguments(output_dir=str(root/'trainer'),use_cpu=True,report_to='none')
            trainer=SentenceTransformerTrainer(model=model,args=args,loss=loss)
            trainer.create_optimizer()
            optimized={id(p) for group in trainer.optimizer.param_groups for p in group['params']}
            self.assertTrue(all(id(p) in optimized for p in loss.head4.parameters()))
            trainer.optimizer.step()
            model.eval();loss.eval()
            with torch.no_grad():
                expected=loss.embeddings(features)
                separate=loss.embeddings(model.tokenize(['three']))[4]
            np.testing.assert_allclose(expected[4][1].numpy(),separate[0].numpy(),atol=1e-6)
            self.assertTrue(torch.allclose(expected[4].norm(dim=-1),torch.ones(2),atol=1e-6))
            loss.save_head(root)
            restored_loss=MultiDepthAlignmentLoss(model,root)
            for a,b in zip(loss.head4.parameters(),restored_loss.head4.parameters()):
                self.assertTrue(torch.equal(a,b))
            exported,cfg=export_exit(model,4,root/'four',{},loss.head4)
            self.assertEqual(len(exported[0].model.transformer.layer),4)
            self.assertEqual(exported[0].model.config.n_layers,4)
            self.assertEqual(len(model[0].model.transformer.layer),6)
            self.assertFalse((root/'four/depth-4-head.safetensors').exists())
            restored=SentenceTransformer(str(root/'four'),device='cpu')
            np.testing.assert_allclose(restored.encode(['one two','three']),expected[4].detach().numpy(),atol=1e-6)
            # Gradients from the shallow exit cannot reach blocks five or six.
            model.zero_grad();loss.zero_grad()
            loss.embeddings(features)[4].sum().backward()
            self.assertIsNone(model[0].model.transformer.layer[5].attention.q_lin.weight.grad)
            self.assertIsNotNone(model[0].model.transformer.layer[0].attention.q_lin.weight.grad)

    def test_depth_defaults_and_invalid_combinations(self):
        self.assertEqual(validate_depths([6],{}),(6,))
        self.assertEqual(validate_depths([4],{'num_layers':4}),(4,))
        self.assertEqual(validate_depths([4,6],{}),(4,6))
        for depths,cfg in [([4],{}),([6],{'num_layers':4}),([6,4],{}),([True],{})]:
            with self.assertRaises(ValueError):validate_depths(depths,cfg)
        for depth in [True,5,'4']:
            with self.assertRaises(ValueError):validate_backend({'num_layers':depth})
        # Re-entering a completed experiment must not erase GPU records or restart its window.
        from unittest.mock import patch
        from scripts.run_multidepth import train
        from foliorecall.io import write_json,read_json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            execution={'started_utc':'2020-01-01T00:00:00Z','window_seconds':14400,
                       'processes':[{'arm':'independent4','returncode':0},{'arm':'joint','returncode':0}],
                       'elapsed_seconds':130.0}
            write_json(root/'execution.json',execution)
            for arm in ('independent4','joint'):write_json(root/f'training/{arm}/result.json',{'complete':True})
            for path in ['training/independent4/final','training/joint/exports/depth-4','training/joint/exports/depth-6']:
                write_json(root/path/'query-config.json',{'num_layers':4 if path.endswith('4') or 'independent4' in path else 6})
            with patch('scripts.run_multidepth.subprocess.run') as child:
                self.assertEqual(train(root),execution)
                child.assert_not_called()
            self.assertEqual(read_json(root/'execution.json'),execution)

    def test_faiss_thread_limit_restores_encoding_threads(self):
        from unittest.mock import patch
        import faiss
        from foliorecall.query import retrieve
        faiss.omp_set_num_threads(4)
        calls=[]
        def encode(*args):
            calls.append(('encode',faiss.omp_get_max_threads()))
            return np.array([[1.,0.]],dtype=np.float32)
        def search(*args):
            calls.append(('search',faiss.omp_get_max_threads()))
            return [[{'page_id':'a'}]]
        with patch('foliorecall.query.encode_query_texts',side_effect=encode),patch('foliorecall.query.search',side_effect=search):
            retrieve(None,{'device':'cpu'},None,[], 'one',search_threads=1)
        self.assertEqual(calls,[('encode',4),('search',1)])
        self.assertEqual(faiss.omp_get_max_threads(),4)

    def test_two_heads_optimizer_and_rng_resume_match_continuous_training(self):
        from copy import deepcopy
        from datasets import Dataset
        from transformers import TrainerCallback
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        class SaveHead(TrainerCallback):
            def __init__(self,loss,stop=False):self.loss,self.stop=loss,stop
            def on_save(self,args,state,control,**kwargs):
                self.loss.save_head(Path(args.output_dir)/f'checkpoint-{state.global_step}')
            def on_step_end(self,args,state,control,**kwargs):
                if self.stop and state.global_step==2:control.should_training_stop=True
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);torch.manual_seed(42);initial=self.tiny(root)
            data=Dataset.from_dict({'query':['one','two','three','one two'],
                                    'label':torch.randn(4,7).tolist()})
            def trainer(model,folder,resume=None,stop=False):
                loss=MultiDepthAlignmentLoss(model,resume)
                args=SentenceTransformerTrainingArguments(output_dir=str(folder),use_cpu=True,
                    max_steps=4,per_device_train_batch_size=2,learning_rate=1e-4,
                    save_strategy='steps',save_steps=2,report_to='none',disable_tqdm=True,seed=42,
                    dataloader_pin_memory=False,remove_unused_columns=False)
                return MultiDepthTrainer(model=model,args=args,train_dataset=data,loss=loss,
                    callbacks=[SaveHead(loss,stop)]),loss
            full,full_loss=trainer(deepcopy(initial),root/'full');full.train()
            split,_=trainer(deepcopy(initial),root/'split',stop=True);split.train()
            checkpoint=root/'split/checkpoint-2'
            resumed,resumed_loss=trainer(SentenceTransformer(str(checkpoint),device='cpu'),root/'split',checkpoint)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            maximum=max(float((x-y).detach().abs().max()) for x,y in zip(full_loss.parameters(),resumed_loss.parameters()))
            print(f'multidepth CPU resume max_abs_diff={maximum:.9g}')
            for (a,x),(b,y) in zip(full_loss.named_parameters(),resumed_loss.named_parameters()):
                self.assertEqual(a,b)
                # CPU kernels can differ at the last FP32 bits after reload.
                torch.testing.assert_close(x,y,rtol=0,atol=1e-6)


if __name__=='__main__':unittest.main()
