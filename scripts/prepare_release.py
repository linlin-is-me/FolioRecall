"""Assemble local, reviewable release assets from retained evidence; never publish."""
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, read_rows, write_json, write_rows
from export_demo_bundle import write_quickstart
from export_stage4_models import add_release_links

TAG = 'v0.1.0rc2'
ROOT = Path('outputs/stage4')
BASE = 'https://github.com/linlin-is-me/FolioRecall'
VERIFICATIONS = ['installed-cpu-rc2-final', 'installed-cpu-post-review',
                 'installed-gpu-query', 'demo-api-gpu', 'browser-gpu',
                 'installed-index-resume-check', 'resume-probe/comparison',
                 'model-export-gpu-final', 'model-export-cli-post-review',
                 'release-materials-final']


def public_result(value):
    """Retain IDs, rankings, metrics and raw timings; do not redistribute task text."""
    if isinstance(value, dict):
        return {key: (len(item) if key == 'queries' and isinstance(item, list)
                      and all(isinstance(query, str) for query in item) else public_result(item))
                for key, item in value.items()
                if key not in {'query', 'text', 'markdown', 'preview'}}
    if isinstance(value, list):
        return [public_result(item) for item in value]
    return value


def archive(folder):
    with zipfile.ZipFile(folder.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as output:
        for file in sorted(folder.rglob('*')):
            if file.is_file():
                output.write(file, file.relative_to(folder.parent))


def record(source, destination, sources, *, verbatim=False):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError(f'必要证据不存在: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == '.json' and not verbatim:
        write_json(destination, public_result(read_json(source)))
    elif source.suffix == '.jsonl' and not verbatim:
        write_rows(destination, public_result(read_rows(source)))
    else:
        shutil.copyfile(source, destination)
    sources.append({'original': str(source), 'published': str(destination),
                    'transformation': 'verbatim' if verbatim or source.suffix not in {'.json', '.jsonl'}
                    else 'omit query/text/markdown/preview fields; query text lists become counts; other numbers and IDs unchanged'})


def run_records(folder, destination, sources):
    # Exact runtime configuration is evidence; preserve paths and diffs as historical.
    for name in ('run.json', 'working-tree.patch', 'process.json'):
        if (folder / name).is_file():
            record(folder / name, destination / name, sources, verbatim=True)
    if (folder / 'untracked').exists():
        for file in sorted((folder / 'untracked').rglob('*')):
            if file.is_file() and file.suffix in {'.py', '.json', '.toml', '.sh', '.cjs'}:
                record(file, destination / 'untracked' / file.relative_to(folder / 'untracked'), sources, verbatim=True)


def prepare(output):
    if output.exists():
        raise ValueError('发布候选已存在，请使用新目录；最终清单使用 --finalize')
    state = provenance(output / 'preparation', {'tag': TAG, 'operation': 'copy retained release assets; no model computation'})
    evidence = output / 'experiment-evidence'
    sources = []
    formal = read_json(ROOT / 'formal-complete/result.json')
    if not formal['complete_matrix'] or len(formal['cells']) != 40:
        raise ValueError('完整40项结果缺失')
    for file in sorted((ROOT / 'formal-complete').glob('*.json')):
        record(file, evidence / 'formal' / file.name, sources)
    record(ROOT / 'formal-complete/results.md', evidence / 'formal/results.md', sources)
    for cell in formal['cells']:
        folder = Path(cell['result']).parent
        destination = evidence / 'evaluations' / cell['task'] / cell['device'] / cell['candidate']
        record(folder / 'result.json', destination / 'result.json', sources)
        run_records(folder, destination, sources)
    for task in read_json('configs/stage4-evaluation.json')['tasks']:
        folder = ROOT / 'bm25' / task['name']
        record(folder / 'result.json', evidence / 'bm25' / task['name'] / 'result.json', sources)
        run_records(folder, evidence / 'bm25' / task['name'], sources)
        # Source metadata carries the fixed dataset revision and historical overlap.
        record(ROOT / 'data' / task['name'] / 'source.json',
               evidence / 'datasets' / task['name'] / 'source.json', sources, verbatim=True)
    for name in VERIFICATIONS:
        folder = ROOT / name
        record(folder / 'result.json', evidence / 'validation' / name / 'result.json', sources)
        run_records(folder, evidence / 'validation' / name, sources)
    for name in ('export-run',):
        run_records(ROOT / 'model-candidates-rc2-final' / name, evidence / 'model-export' / name, sources)
    record('outputs/stage2/teacher.json', evidence / 'teacher.json', sources, verbatim=True)
    for file in sorted(Path('configs').glob('*.json')):
        record(file, evidence / 'configs' / file.name, sources, verbatim=True)
    # Retain build records without distributing indexes or their page images.
    for item in formal['indexes']:
        folder = Path(item['index'])
        destination = evidence / 'indexes' / item['task'] / item['teacher']
        for name in ('config.json', 'build.json'):
            if (folder / name).exists():
                record(folder / name, destination / name, sources, verbatim=True)
        run_records(folder, destination, sources)
    for source in sources:
        source['published'] = str(Path(source['published']).relative_to(evidence))
    write_json(evidence / 'sources.json', sources)
    (evidence / 'README.md').write_text(
        '# FolioRecall 实验附件\n\nformal为五域汇总及三组逐查询比较，evaluations保留40项神经检索的'
        '逐查询排名、指标和原始逐请求计时，bm25为五域文本对照。以(task, query_id)配对。'
        '查询原文、原标签和页面从configs/stage4-evaluation.json固定revision获取；附件不附训练文本或页面图像。\n\n'
        'sources.json给出原本机位置与本附件相对位置。run.json和working-tree.patch是原运行版本与差异，'
        '其中绝对路径、索引和检查点目录是历史记录，不是新机器的前置路径。'
        'teacher.json保留选定教师原记录；cache-teacher读取其内嵌编码配置，旧index字段仅供追溯。'
        '可搬移复现命令见源码doc/首版使用与评测.md；configs为发布时仓库配置，历史生效配置以各run.json为准。\n\n'
        'GPU与恢复证据直接复用既有运行；本轮仅追加CPU安装和页面展示验证。历史原HR建库缺失字段仍缺失。'
        '本项目结果与代码采用Apache-2.0，上游数据许可不因该附件改变。\n', encoding='utf-8')
    shutil.copyfile('LICENSE', evidence / 'LICENSE')
    shutil.copyfile('NOTICE', evidence / 'NOTICE')
    shutil.copytree(ROOT / 'demo-bundle', output / 'demo-bundle')
    bundle = output / 'demo-bundle'
    write_quickstart(bundle)
    metadata = read_json(bundle / 'bundle.json')
    metadata.update(status='materials reviewed; local release candidate; publication pending',
                    materials_evidence='experiment-evidence/validation/release-materials-final/result.json',
                    previous_bundle='outputs/stage4/demo-bundle', release_tag=TAG)
    write_json(bundle / 'bundle.json', metadata)
    text = (bundle / 'SOURCES.md').read_text(encoding='utf-8')
    (bundle / 'SOURCES.md').write_text(text.replace('公开上传前须核对最终素材。',
        '当前42页范围已完成本地素材核对，见实验附件release-materials-final；不扩大第三方内容许可。'), encoding='utf-8')
    archive(bundle)
    for name in ('lora750', 'distilled94'):
        shutil.copytree(ROOT / 'model-candidates-rc2-final' / name, output / name)
        add_release_links(output / name, TAG)
        info = read_json(output / name / 'source.json')
        info.update(release_preparation_commit=state['commit'],
                    original_export_commit=read_json(ROOT / 'model-candidates-rc2-final/export-run/run.json')['commit'],
                    payload='copied unchanged from previously verified rc2-final package')
        write_json(output / name / 'source.json', info)
        archive(output / name)
    (output / 'release-notes.md').write_text(
        '# FolioRecall 0.1.0rc2 预发布\n\n英文页面检索：PDF导入、可恢复页面建库、独立查询学生、常驻本地页面演示。'
        '验证系统为Python 3.10、Ubuntu/WSL，GPU为RTX 4060 Laptop 8GB；默认原始Qwen教师建库、公开ML查询，CPU仅需学生即可体验。\n\n'
        '五领域完整12969页、1489查询，25项GPU及15项CPU神经检索与五域BM25。'
        'GPU宏平均nDCG@10：原始0.537845，LoRA750 0.573712，公开EN 0.566762，公开ML 0.567706，继续蒸馏第94步0.437830。'
        'LoRA四域提高、Finance退化；ML同卡BF16编码P50快6.23–7.65倍，公开模型收益归属NanoVDR上游。'
        '第94步仅作为无收益对照，不推荐替换默认ML。\n\n'
        f'安装与CPU体验见[README]({BASE}/blob/{TAG}/README.md)，'
        f'GPU自有PDF与完整复现见[使用说明]({BASE}/blob/{TAG}/doc/首版使用与评测.md)。'
        'wheel与源码包二选一；CPU体验另下载demo-bundle.zip。lora750.zip和distilled94.zip为可选实验模型，'
        'experiment-evidence.zip保存结果与实际运行依据，release-manifest.json记录资产及构建版本。\n\n'
        '代码Apache-2.0；模型和欧盟示例保留来源及各自许可，不附完整PDF、教师基座或训练数据。'
        '尚无中文评测、无答案检测或多用户服务。结果为单seed固定任务证据，上游重叠未知。'
        '安装和实际下载地址的核实状态以release-manifest及开发计划为准。\n', encoding='utf-8')


def finalize(output):
    evidence = output / 'experiment-evidence'
    validation = read_json(output / 'validation/result.json')
    if not validation.get('passed'):
        raise ValueError('尚未通过本轮CPU安装和展示验证')
    sources = read_json(evidence / 'sources.json')
    for file in sorted((output / 'validation').rglob('*')):
        if not file.is_file() or file.suffix not in {'.json', '.jsonl', '.png', '.patch', '.py', '.toml', '.sh', '.cjs'}:
            continue
        destination = evidence / 'validation/release-cpu' / file.relative_to(output / 'validation')
        record(file, destination, sources)
        sources[-1]['published'] = str(destination.relative_to(evidence))
    extra = []
    run_records(output / 'build', evidence / 'package-build', extra)
    run_records(output / 'preparation', evidence / 'release-preparation', extra)
    run_records(output / 'preparation/query-text-correction', evidence / 'release-preparation/query-text-correction', extra)
    for item in extra:
        item['published'] = str(Path(item['published']).relative_to(evidence))
    write_json(evidence / 'sources.json', sources + extra)
    archive(evidence)
    names = ['foliorecall-0.1.0rc2-py3-none-any.whl', 'foliorecall-0.1.0rc2.tar.gz',
             'demo-bundle.zip', 'lora750.zip', 'distilled94.zip', 'experiment-evidence.zip']
    assets = [{'name': name, 'bytes': (output / name).stat().st_size,
               'planned_url': f'{BASE}/releases/download/{TAG}/{name}'} for name in names]
    write_json(output / 'release-manifest.json', {
        'version': '0.1.0rc2', 'tag': TAG, 'status': 'local validated candidate; remote publication not authorized',
        'build_commit': read_json(output / 'build/run.json')['commit'],
        'assets': assets, 'manifest_asset': 'release-manifest.json',
        'validation': 'experiment-evidence/validation/release-cpu/result.json',
        'release_notes': 'release-notes.md', 'gpu_work_this_round': False,
        'remote_download_validation': 'pending publication authorization',
        'publication_scope': 'main and annotated tag; GitHub prerelease with these six assets and this manifest; no PyPI or HF upload'})
    print(output / 'release-manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', default='outputs/stage4/release-v0.1.0rc2')
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    (finalize if args.finalize else prepare)(Path(args.output))
