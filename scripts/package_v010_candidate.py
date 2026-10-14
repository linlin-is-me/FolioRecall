"""Assemble the local v0.1.0 review candidate from already verified artifacts.

Run from the checkout with the existing CPU environment. This script neither
downloads models nor exports graphs, encodes pages, or contacts a generation API.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/releases/v0.1.0-final-candidate'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def documents(out):
    package = out / 'package'
    for name in ('LICENSE', 'NOTICE', '.env.example'):
        shutil.copy2(ROOT / name, package / name)
    shutil.copy2(ROOT / 'doc/v0.1.0快速开始.md', package / 'QUICKSTART.md')
    release = (ROOT / 'doc/v0.1.0发布说明.md').read_text(encoding='utf-8')
    start = release.index('## 审核入口')
    release = release[:start] + '''## 审核入口

从 [QUICKSTART](QUICKSTART.md) 启动，查看 [验证摘要](VALIDATION.md)、
[附件](ASSETS.md) 和 screenshots 中的真实界面。完整双语 README、实验结果和
项目介绍位于源码包的 README 与 doc 目录。

这是本地审核候选，源码推送、标签和 GitHub Release 等待用户确认。
'''
    release = release.replace('outputs/releases/v0.1.0-final-candidate/package', '当前附件目录')
    release = release.replace('[ONNX记录](首版使用与评测.md#onnx-cpu-backends)', '源码包中的 ONNX 记录')
    (package / 'RELEASE-NOTES.md').write_text(release, encoding='utf-8')
    for source, target in (('validation-summary.json', 'validation-summary.json'),
                           ('VALIDATION.md', 'VALIDATION.md')):
        if (out / source).exists():
            shutil.copy2(out / source, package / target)
    base = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip()
    changes = subprocess.check_output(['git', 'diff', '--stat', 'origin/main', '--'], cwd=ROOT, text=True)
    (package / 'CHANGE-SUMMARY.md').write_text(
        '# 相对远端 main 的变化\n\n'
        f'比较基点：`{base}`（本地远端跟踪记录）。\n\n'
        '包含双语 RAG、受限 Agent、多轮追问、实时步骤、完整 ONNX 查询后端，'
        '以及可复用训练与检索对照。实验入口保留，未采用方案不进入产品默认。\n\n'
        '```text\n' + changes + '```\n\n尚未推送、打标签或创建 Release。\n', encoding='utf-8')
    files = sorted(p for p in package.rglob('*') if p.is_file() and p.name != 'ASSETS.md')
    (package / 'ASSETS.md').write_text(
        '# 候选附件\n\n软件与模型分开分发；模型不进入 wheel、源码包或 Git。'
        '不含完整 PDF、凭据、人工私有映射或原始运行目录。\n\n'
        '| 文件 | 字节 |\n|---|---:|\n' + ''.join(
            f'| {p.relative_to(package).as_posix()} | {p.stat().st_size} |\n' for p in files), encoding='utf-8')
    archive = out / 'foliorecall-0.1.0-final-review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():
                z.write(p, Path('foliorecall-0.1.0-final-review') / p.relative_to(package))
    print('Local candidate:', package, flush=True)


def prepare(out):
    if out.exists():
        raise ValueError('目标已存在；保留旧记录，改用新目录或只刷新文档')
    package = out / 'package'
    package.mkdir(parents=True)
    old = ROOT / 'outputs/releases/v0.1.0-candidate'
    onnx = ROOT / 'outputs/extensions/D/onnx-int8-v1/candidate/package'
    demo = out / 'demo-bundle'
    shutil.copytree(old / 'demo-bundle', demo)
    for name in ('baseline.json', 'agent-demo.json', 'student-ml-cpu.json',
                 'student-ml-cpu-default.json', 'student-ml-onnx-fp32.json', 'student-ml-onnx-int8.json'):
        shutil.copy2(ROOT / 'configs' / name, demo / 'configs' / name)
        (package / 'configs').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'configs' / name, package / 'configs' / name)
    shutil.copy2(ROOT / 'doc/v0.1.0快速开始.md', demo / 'QUICKSTART.md')
    with zipfile.ZipFile(package / 'demo-bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(demo.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(demo.parent))
    for name in ('query-onnx-fp32.zip', 'query-onnx-int8.zip'):
        shutil.copy2(onnx / name, package / name)
    (package / 'screenshots').mkdir()
    for name in ('v010-agent-followup.png', 'v010-agent-running.png', 'v010-insufficient.png',
                 'onnx-fp32-retrieval.png'):
        shutil.copy2(ROOT / 'doc/assets' / name, package / 'screenshots' / name)
    write(out / 'assembly.json', {
        'version': '0.1.0', 'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'demo_source': str(old.relative_to(ROOT)), 'model_source': str(onnx.relative_to(ROOT)),
        'query_backend': 'onnx-fp32', 'pages': 42,
        'models_exported': False, 'pages_reencoded': False, 'generation_api_calls': 0,
        'historical_candidates_preserved': True})
    subprocess.run([sys.executable, '-m', 'build', '--no-isolation', '--outdir', str(package)],
                   cwd=ROOT, check=True)
    documents(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--refresh-docs', action='store_true', help='Refresh docs and source archive after validation')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.refresh_docs:
        if not (out / 'assembly.json').exists():
            raise ValueError('先完成候选构建')
        subprocess.run([sys.executable, '-m', 'build', '--sdist', '--no-isolation',
                        '--outdir', str(out / 'package')], cwd=ROOT, check=True)
        documents(out)
    else:
        prepare(out)


if __name__ == '__main__':
    main()
