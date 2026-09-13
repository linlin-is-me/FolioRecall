"""Local release candidates only; no upload and no optimizer/training-data export."""
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json


def add_release_links(target, tag):
    """Annotate a copied model package without changing its loading configuration."""
    target = Path(target)
    base = 'https://github.com/linlin-is-me/FolioRecall'
    source = read_json(target / 'source.json')
    source.update(release_tag=tag,
                  results_url=f'{base}/blob/{tag}/doc/首版使用与评测.md',
                  evidence_url=f'{base}/releases/download/{tag}/experiment-evidence.zip',
                  historical_paths='checkpoint, formal_summary and export_run identify original local evidence; use evidence archive sources.json for portable references')
    write_json(target / 'source.json', source)
    with (target / 'README.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n## 公开获取与证据\n\n目标版本：{tag}，实际发布状态以'
                     f'[Release]({base}/releases/tag/{tag})为准。'
                     f'[完整结果]({source["results_url"]})；'
                     f'[实验附件]({source["evidence_url"]})包含实际运行版本、生效配置与加载验证记录。'
                     'source.json中的旧路径用于追溯历史，不要求在新机器创建同名目录。\n')


def export(output, summary=None):
    output = Path(output)
    if output.exists():
        raise ValueError('模型候选目录已存在，请使用新目录')
    sources = {'lora750': Path('outputs/stage2/lora/checkpoint-750'),
               'distilled94': Path('outputs/stage3/distill-3000/checkpoint-94')}
    provenance(output / 'export-run', {'checkpoints': {k: str(v.resolve()) for k, v in sources.items()}, 'formal_summary': summary})
    teacher = read_json('outputs/stage2/teacher.json')
    formal = read_json(summary) if summary else None
    for name, source in sources.items():
        target = output / name
        target.mkdir()
        required = ['modules.json', 'config_sentence_transformers.json', 'sentence_bert_config.json',
                    'tokenizer_config.json', 'tokenizer.json']
        required += ['adapter_config.json', 'adapter_model.safetensors', 'processor_config.json', 'chat_template.jinja'] if name == 'lora750' else ['config.json', 'model.safetensors']
        for file in required:
            shutil.copyfile(source / file, target / file)
        for module in read_json(source / 'modules.json'):
            if module['path']:
                shutil.copytree(source / module['path'], target / module['path'])
        # Trainer-generated cards contain example training queries and local image
        # paths. Keep those in the experiment checkpoint, not the release package.
        shutil.copyfile('LICENSE', target / 'LICENSE')
        (target / 'UPSTREAM.md').write_text(
            '# 来源与许可\n\n'
            '- Qwen3-VL-Embedding-2B：[固定版本模型卡](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/'+teacher['revision']+'/README.md)，Apache-2.0。\n'
            '- NanoVDR公开ML：[固定版本模型卡](https://huggingface.co/nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML/blob/ab3f0fde9fcf407eaa756e1fc349ac09b7a716e7/README.md)，Apache-2.0。\n'
            '- 训练接口复用 [Sentence Transformers](https://github.com/huggingface/sentence-transformers)、[Transformers](https://github.com/huggingface/transformers) 和 [PEFT](https://github.com/huggingface/peft)。\n'
            '- 本项目使用 [llamaindex/vdr-multilingual-train](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train) 的冻结3000条英文训练记录。原数据及相关许可不随模型包转授，包内不分发查询或页面。\n\n'
            'FolioRecall的改动是本地LoRA训练或继续查询蒸馏，以及完整加载配置；不是新骨干或上游方法的原创实现。原检查点内训练器生成的模型卡不属于上游许可文件，因含训练示例和本机路径，不纳入本包。\n', encoding='utf-8')
        if name == 'lora750':
            config = read_json(source / 'encoding.json')
            config['adapter'] = '.'
            write_json(target / 'encoding.json', config)
            instructions = '在本包目录运行 `foliorecall index --config encoding.json --pages <清单> --output <新库>`，再用 `foliorecall query --config encoding.json --index <新库> <查询> --json` 查询。需要安装页面与适配器依赖 `foliorecall[image,train]`。adapter相对当前工作目录解析。它不能与原始教师页面库混用，已生成实验索引的本机绝对路径配置不直接改写。'
            findings = '内部开发nDCG@10从原始0.969236到0.977737；完整HR GPU为0.542759，原始教师为0.518239。默认页面教师仍为原始模型，不按单个领域结果更换。'
        else:
            config = read_json(source / 'query-config.json')
            config['model_path'] = '.'
            write_json(target / 'query-config.json', config)
            write_json(target / 'query-gpu.json', config)
            write_json(target / 'query-cpu.json', dict(config, device='cpu', dtype='float32'))
            instructions = '在本包目录运行 `foliorecall query --config <原始教师配置> --query-config query-cpu.json --index <匹配原始教师库> <查询> --json`；GPU使用query-gpu.json。model_path相对当前工作目录解析，包内骨干、tokenizer、pooling、两层投影和Normalize一起加载。在线不需要教师权重。'
            findings = '3000查询余弦继续蒸馏未改善检索：内部nDCG@10为0.953472，低于公开ML的0.972699；完整HR CPU为0.453054，低于公开ML的0.542341。仅保留作实验对照，默认仍为公开ML。'
        if formal:
            candidate = formal['macro'].get('gpu/' + name)
            if candidate:
                findings += f" 五领域完整GPU宏平均nDCG@10为{candidate['nDCG@10']:.6f}；固定任务结果，不证明文档独立或上游训练无重叠。"
        base = 'Qwen/Qwen3-VL-Embedding-2B' if name == 'lora750' else 'nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML'
        (target / 'README.md').write_text(f'---\nlicense: apache-2.0\nbase_model: {base}\n---\n\n# FolioRecall {name} 本地待发布候选\n\n{findings}\n\n{instructions}\n\n'
            '页面教师基础模型：Qwen/Qwen3-VL-Embedding-2B，revision '+teacher['revision']+'。查询学生的公开ML初始化：nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML，revision ab3f0fde9fcf407eaa756e1fc349ac09b7a716e7；LoRA750本身从Qwen教师初始化。\n\n'
            '项目代码Apache-2.0不变更上游模型或训练数据许可。上游来源及声明见UPSTREAM.md，许可文本见LICENSE。训练数据不随包分发，上游训练及文档级重叠尚未完全核实。\n\n'
            '本次只复制加载所需文件和来源说明，不包含优化器、训练数据或完整教师权重。加载验证结果单独保存在阶段四实验记录，导出动作本身不代表验证通过或已经公开发布。\n', encoding='utf-8')
        write_json(target / 'source.json', {'checkpoint': str(source), 'teacher_revision': teacher['revision'],
            'status': 'local candidate; package validation recorded separately; publication pending',
            'formal_summary': summary,
            'export_run': str(output / 'export-run/run.json')})
        with zipfile.ZipFile(output / f'{name}.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for file in target.rglob('*'):
                if file.is_file():
                    archive.write(file, file.relative_to(output))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/stage4/model-candidates-rc2')
    parser.add_argument('--summary', help='Completed formal result.json for model-card metrics')
    args = parser.parse_args()
    export(args.output, args.summary)
