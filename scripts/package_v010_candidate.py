"""Assemble the local v0.1.0 review candidate from already verified artifacts.

Run from the checkout with the existing CPU environment. This script neither
downloads models nor exports graphs, encodes pages, or contacts a generation API.
"""
import argparse
import json
from pathlib import Path
import re
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
    if json.loads((out/'assembly.json').read_text()).get('workflow') == 'research':
        research_documents(out)
        return
    package = out / 'package'
    (package / 'branding').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'doc/assets/foliorecall-wordmark.png', package / 'branding/foliorecall-wordmark.png')
    shutil.copy2(ROOT / 'doc/assets/foliorecall-lettermark.png', package / 'branding/foliorecall-lettermark.png')
    shutil.copy2(ROOT / 'foliorecall/assets/wordmark.png', package / 'branding/wordmark.png')
    shutil.copy2(ROOT / 'foliorecall/assets/logo.png', package / 'branding/logo-256.png')
    shutil.copy2(ROOT / 'doc/assets/logo-source.md', package / 'branding/README.md')
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
    if (out / 'validation/final-retrieval.jpg').exists():
        shutil.copy2(out / 'validation/final-retrieval.jpg', package / 'screenshots/final-retrieval.jpg')
    for source, name in (('v010-retrieval-ui.jpg', 'ui-desktop.jpg'),
                         ('v010-chat-layout.jpg', 'ui-chat.jpg')):
        if (ROOT / 'doc/assets' / source).exists():
            shutil.copy2(ROOT / 'doc/assets' / source, package / 'screenshots' / name)
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


RESEARCH_CONFIGS = ('baseline.json', 'agent-demo.json', 'agent-multi-demo.json',
    'agent-research-demo.json', 'student-ml-cpu.json', 'student-ml-cpu-default.json',
    'student-ml-onnx-fp32.json', 'student-ml-onnx-int8.json')


def research_documents(out):
    """Refresh portable navigation and the manifest; preserve measured results."""
    package = out/'package'
    record = json.loads((out/'assembly.json').read_text())
    showcase = ROOT/'outputs/extensions/A/agent-showcase-v1'
    has_showcase = (showcase/'index.html').exists()
    if has_showcase:
        # Keep the public cases separate from local verification logs and fixtures.
        target = package/'showcase'
        target.mkdir(exist_ok=True)
        for name in ('index.html', 'SOURCES.md', 'manifest.json'):
            shutil.copy2(showcase/name, target/name)
        for name in ('cases', 'pages'):
            shutil.copytree(showcase/name, target/name, dirs_exist_ok=True)
        record['showcase_source'] = str(showcase.relative_to(ROOT))
        record['showcase_cases'] = json.loads((showcase/'manifest.json').read_text(encoding='utf-8'))['cases']
    for name in ('SHOWCASE-VALIDATION.md', 'showcase-validation.json'):
        if (out/name).exists():
            shutil.copy2(out/name, package/name)
    if (ROOT/'doc/assets/agent-showcase-research.jpg').exists():
        shutil.copy2(ROOT/'doc/assets/agent-showcase-research.jpg', package/'screenshots/agent-showcase-research.jpg')
    showcase_intro = ('''本次新增实时协作状态、任务责任与依赖、原页归属和版本化核验展示。
四份离线案例复用历史真实记录，在当前浅色界面重新呈现，不重新生成答案。
本轮模拟生成器的浏览器验证仅检查界面与装配，不作为新的真实模型质量或性能结果。
发布前修订进一步区分历史真实验证与当前模拟装配。取证栏表示工具成功读页，
不代表该取证模型随后收到过图像；分析接收版本与最终引用另列。
未解决事项来自模型报告，不保证枚举全部遗漏。同次尝试读取检查点，独立补跑
复用成功操作；会话由服务端Gradio内存状态保存。本次仅修正文案和标签。
''' if has_showcase else '''本次修订示例模式、导航、打包和说明，没有重新生成实验回答。
''')
    showcase_links = ('- [四份离线协作案例](showcase/index.html)：无需API密钥、模型或索引。\n'
                      '- [案例来源与范围](showcase/SOURCES.md)。\n' if has_showcase else '')
    if (package/'SHOWCASE-VALIDATION.md').exists():
        showcase_links += '- [本轮展示与安装验证](SHOWCASE-VALIDATION.md)。\n'
    if (package/'showcase-validation.json').exists():
        showcase_links += '- [本轮结构化验证记录](showcase-validation.json)。\n'
    (package/'RELEASE-NOTES.md').write_text('''# FolioRecall v0.1.0 文档研究候选

本地审核候选，尚未推送或发布。规划协调、责任取证、分析和核验组成研究流程，
按依赖调度，最多一次定向返工；单Agent继续用于快速问答。

60次真实对照及助手评审、六次安装态浏览器确认保留原结果。
研究模式整体覆盖未优于单Agent，评分保持human=not_checked。
'''+showcase_intro+'''界面默认使用白色卡片、浅灰背景和绿色强调色，问答方法保持不变。
本轮没有新增付费调用、模型导出、页面编码或质量评测。

## 启动与核对

- [快速开始](QUICKSTART.md)：安装、免费检索、研究模式与快速问答。
'''+showcase_links+'''
- [实测结果与限制](RESULTS.md)和[结果摘要](results-summary.json)。
- [历史研究候选验证](VALIDATION.md)和[历史六次真实浏览器确认](browser-validation.json)。
- [附件清单](ASSETS.md)与[变更摘要](CHANGE-SUMMARY.md)。
- [源码包](foliorecall-0.1.0.tar.gz)：解压后查看README.md、README.zh-CN.md与doc/中的面试材料及完整实验说明。

模型和42页示例单独提供。完整PDF、教师权重、凭据、原始运行和检查点不在附件内。
JRC案例来自本机256页库；离线案例仅附精选页面，不提供完整JRC库。
历史实测、当前展示及便携42页库的功能验证分别记录。
''',encoding='utf-8')
    if (ROOT/'doc/assets/research-light-ui.png').exists():
        shutil.copy2(ROOT/'doc/assets/research-light-ui.png', package/'screenshots/research-light-ui.png')
    quick = (ROOT/'doc/v0.1.0快速开始.md').read_text(encoding='utf-8')
    if has_showcase and 'showcase/index.html' not in quick:
        title, body = quick.split('\n', 1)
        quick = title+'''\n\n## 先看协作案例，无需API

直接打开候选附件目录中的 `showcase/index.html`，查看四份历史真实调用的任务、
原页、草稿和核验记录。无需安装Python、加载模型或填写API密钥。
案例的当前浅色展示没有触发新调用；模拟生成器验证也不计入原60次真实结果。
需要执行免费检索或发起新问答时，再按下文安装。
'''+body
    # This copy is also embedded in the demo zip, which may be extracted elsewhere.
    quick = re.sub(r'\[([^\]]+)\]\(首版使用与评测\.md#[^)]+\)',
                   r'\1（见附件RESULTS.md及源码包doc/首版使用与评测.md）',quick)
    (package/'QUICKSTART.md').write_text(quick,encoding='utf-8')
    bundle = package/'demo-bundle.zip'
    temporary = out/'demo-bundle-refresh.zip'
    replacements = {'demo-bundle/QUICKSTART.md':quick.encode('utf-8')}
    replacements.update({'demo-bundle/configs/'+name:(ROOT/'configs'/name).read_bytes()
                         for name in RESEARCH_CONFIGS})
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            if item.filename not in replacements:
                target.writestr(item,source.read(item.filename))
        for name,data in replacements.items():
            target.writestr(name,data)
    temporary.replace(bundle)
    for name in RESEARCH_CONFIGS:
        shutil.copy2(ROOT/'configs'/name,package/'configs'/name)
    base = subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    changes = subprocess.check_output(['git','diff','--stat','origin/main','--'],cwd=ROOT,text=True)
    (package/'CHANGE-SUMMARY.md').write_text(
        f'# 候选来源\n\nwheel源码：`{record["source_commit"]}`；付费方法：`{record["paid_runtime_commit"]}`。\n\n'
        f'源码包及文档：`{record.get("sdist_source_commit", record["source_commit"])}`。\n\n'
        f'比较基点：本地缓存origin/main `{base}`，未刷新远端。\n\n'+
        ('问答方法和原始实验保留。本轮增加实际事件驱动的协作视图及历史案例展示，'
         '没有新增模型调用或质量评测。\n\n' if has_showcase else
         '研究方法与原始实验保留，本轮更新便携导航与说明。\n\n')+
        '```text\n'+changes+'```\n',encoding='utf-8')
    files = sorted(p for p in package.rglob('*') if p.is_file() and p.name!='ASSETS.md')
    (package/'ASSETS.md').write_text(
        '# 本地审核附件\n\n示例来源见demo-bundle.zip中的SOURCES.md，模型来源及许可随独立模型包提供。'
        'RESULTS.md、results-summary.json与browser-validation.json保留历史真实实验。'
        'VALIDATION.md为继承的历史验证记录，不表示当前组合再次执行了付费问答。\n\n'+
        ('[离线协作案例](showcase/index.html)使用历史真实记录，来源见[案例说明](showcase/SOURCES.md)。'
         'agent-showcase-research.jpg为这些记录在当前浅色界面的实际截图；'
         '其余研究截图保留历史展示。本轮模拟界面与安装检查见SHOWCASE-VALIDATION.md及'
         'showcase-validation.json（有实际记录时提供），不作为新质量结果。\n\n'
         if has_showcase else '')+
        '| 文件 | 字节 |\n|---|---:|\n'+''.join(
            f'| [{p.relative_to(package).as_posix()}]({p.relative_to(package).as_posix()}) | {p.stat().st_size} |\n'
            for p in files),encoding='utf-8')
    archive = out/'foliorecall-0.1.0-research-review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file(): z.write(p,Path('package')/p.relative_to(package))
    write(out/'assembly.json',dict(record,files=len(files)+1,archive_bytes=archive.stat().st_size,
                                  archive=archive.name))
    print('Local research candidate:',package,flush=True)


def prepare_research(out, source):
    """Reuse reviewed assets; build code and update the eight portable configs."""
    if out.exists():
        raise ValueError('目标已存在；保留旧候选，使用新的输出目录')
    package = out/'package'
    package.mkdir(parents=True)
    (package/'configs').mkdir()
    for name in ('demo-bundle.zip','query-onnx-fp32.zip','query-onnx-int8.zip',
                 'results-summary.json','RESULTS.md','browser-validation.json','VALIDATION.md'):
        shutil.copy2(source/name,package/name)
    for name in ('screenshots','branding'):
        shutil.copytree(source/name,package/name)
    for name in ('LICENSE','NOTICE','.env.example'):
        shutil.copy2(ROOT/name,package/name)
    if (source/'final-install-validation.json').exists():
        shutil.copy2(source/'final-install-validation.json',package/'previous-install-validation.json')
    inherited = (package/'VALIDATION.md').read_text(encoding='utf-8')
    (package/'VALIDATION.md').write_text(
        '# 继承的历史研究候选验证\n\n以下内容及browser-validation.json来自来源候选，'
        '对应原有60次真实结果和历史安装验证，不表示本轮重新调用生成模型。'
        '本轮展示与安装检查如已完成，另存SHOWCASE-VALIDATION.md和showcase-validation.json。'
        'previous-install-validation.json仅在来源候选提供时保留。\n\n---\n\n'+inherited,
        encoding='utf-8')
    results = json.loads((package/'results-summary.json').read_text())
    write(out/'assembly.json',{'version':'0.1.0','workflow':'research',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'paid_runtime_commit':results['runtime_commit'],'asset_source':str(source),
        'query_backend':'onnx-fp32','pages':42,'models_exported':False,
        'pages_reencoded':False,'generation_api_calls':0,'historical_candidates_preserved':True})
    subprocess.run([sys.executable,'-m','build','--no-isolation','--outdir',str(package)],cwd=ROOT,check=True)
    research_documents(out)


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
    parser.add_argument('--research-source', type=Path,
                        help='复用已评审研究候选的package目录，构建新的研究候选')
    parser.add_argument('--refresh-docs', action='store_true', help='Refresh docs and source archive after validation')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.refresh_docs:
        if not (out / 'assembly.json').exists():
            raise ValueError('先完成候选构建')
        subprocess.run([sys.executable, '-m', 'build', '--sdist', '--no-isolation',
                        '--outdir', str(out / 'package')], cwd=ROOT, check=True)
        record = json.loads((out/'assembly.json').read_text())
        record['sdist_source_commit'] = subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(out/'assembly.json',record)
        documents(out)
    elif args.research_source:
        prepare_research(out,args.research_source.resolve())
    else:
        prepare(out)


if __name__ == '__main__':
    main()
