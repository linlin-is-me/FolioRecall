"""Local release candidates only; no upload and no optimizer/training-data export."""
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json


def export(output):
    output = Path(output)
    if output.exists():
        raise ValueError('模型候选目录已存在，请使用新目录')
    sources = {'lora750': Path('outputs/stage2/lora/checkpoint-750'),
               'distilled94': Path('outputs/stage3/distill-3000/checkpoint-94')}
    provenance(output / 'export-run', {k: str(v.resolve()) for k, v in sources.items()})
    teacher = read_json('outputs/stage2/teacher.json')
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
        shutil.copyfile(source / 'README.md', target / 'UPSTREAM_GENERATED_MODEL_CARD.md')
        if name == 'lora750':
            config = read_json(source / 'encoding.json')
            config['adapter'] = '.'
            write_json(target / 'encoding.json', config)
            instructions = '在本包目录运行 index --config encoding.json --pages <清单> --output <新库>，再用同一配置查询该库。adapter相对当前工作目录解析。它不能与原始教师页面库混用，已生成实验索引的本机绝对路径配置不直接改写。'
            findings = '内部开发nDCG@10从原始0.969236到0.977737；未验证外部泛化，不是已确认优于原始教师的默认模型。'
        else:
            config = read_json(source / 'query-config.json')
            config['model_path'] = '.'
            write_json(target / 'query-gpu.json', config)
            write_json(target / 'query-cpu.json', dict(config, device='cpu', dtype='float32'))
            instructions = '在本包目录运行 query --config <原始教师配置> --query-config query-cpu.json --index <匹配原始教师库> <查询>；GPU使用query-gpu.json。model_path相对当前工作目录解析，包内骨干、tokenizer、pooling、两层投影和Normalize一起加载。在线不需要教师权重。'
            findings = '3000查询余弦继续蒸馏未改善检索：内部nDCG@10为0.953472，低于公开ML的0.972699；完整HR CPU为0.453054，低于公开ML的0.542341。仅保留作实验对照，默认仍为公开ML。'
        (target / 'README.md').write_text(f'# FolioRecall {name} 本地待发布候选\n\n{findings}\n\n{instructions}\n\n'
            '页面教师基础模型：Qwen/Qwen3-VL-Embedding-2B，revision '+teacher['revision']+'。查询学生的公开ML初始化：nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML，revision ab3f0fde9fcf407eaa756e1fc349ac09b7a716e7；LoRA750本身从Qwen教师初始化。\n\n'
            '项目代码Apache-2.0不变更上游模型或训练数据许可。上游来源及声明保存在UPSTREAM_GENERATED_MODEL_CARD.md；公开上传前核对模型许可和最终模型卡。训练数据不随包分发，上游训练及文档级重叠尚未完全核实。\n\n'
            '本次只复制加载所需文件和来源说明，不包含优化器、训练数据或完整教师权重。原检查点已验证，本次新包的GPU加载待资源时段验证；未上传模型。\n', encoding='utf-8')
        write_json(target / 'source.json', {'checkpoint': str(source), 'teacher_revision': teacher['revision'],
            'status': 'local candidate; GPU package validation and publication pending',
            'export_run': str(output / 'export-run/run.json')})
        with zipfile.ZipFile(output / f'{name}.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for file in target.rglob('*'):
                if file.is_file():
                    archive.write(file, file.relative_to(output))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/stage4/model-candidates-rc2')
    args = parser.parse_args()
    export(args.output)
