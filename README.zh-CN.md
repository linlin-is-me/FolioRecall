<h1 align="center">FolioRecall · 页寻</h1>
<p align="center"><strong>视觉文档检索、双语RAG与有限Agent补查</strong></p>
<p align="center">检索文档中的文字、表格与图表，返回原页证据、物理页码和结构化回答。</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-607d8b?style=flat" alt="许可：Apache-2.0"></a>
  <a href="#快速开始"><img src="https://img.shields.io/badge/Tested_Python-3.10-607d8b?style=flat" alt="已验证 Python：3.10"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-2.8-607d8b?style=flat" alt="PyTorch：2.8"></a>
</p>
<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>
<p align="center"><a href="#演示">演示</a> · <a href="#功能">功能</a> · <a href="#快速开始">快速开始</a> · <a href="#结果">结果</a> · <a href="#个人实现与上游来源">贡献</a></p>

FolioRecall先检索页面证据，再执行固定RAG或有限Agent问答。多模态教师离线建库，轻量查询学生在线搜索；可选页面BM25与视觉结果融合。英文和中文问句均可检索英文文档，回答提供可跳转的原页引用和JSON。

> 源码已公开。`0.1.0rc3`、42页示例包和发布说明已在本地整理，附件尚未公开下载。

## 演示

42页示例的真实CPU检索命中人口报告的**物理第10页、Figure 3**，并展示来源、预览和JSON。

<p align="center">
  <a href="doc/assets/stage4-cpu-demo.png"><img src="doc/assets/stage4-cpu-demo.png" width="640" alt="真实CPU页面检索，显示来源、物理页码、页面预览与JSON下载。"></a>
</p>

<details>
<summary>查看真实RAG与Agent问答截图</summary>

第9题Hybrid RAG展示回答与原页引用；第22题中文Hybrid Agent展示跨文档补查。截图来自已保存的真实使用记录，未为展示重新生成回答。

<a href="doc/assets/a6-rag-jrc09.png"><img src="doc/assets/a6-rag-jrc09.png" width="600" alt="第9题英文Hybrid RAG实际回答和页面引用。"></a>
<a href="doc/assets/a6-agent-jrc22.png"><img src="doc/assets/a6-agent-jrc22.png" width="600" alt="第22题中文Hybrid Agent实际回答和跨文档页面证据。"></a>

</details>

[五分钟演示与不足回答案例](doc/项目介绍与面试准备.md#五分钟演示) · [安装态真实使用结果](doc/首版使用与评测.md#a6-收尾计划与展示目标)

## 功能

- **页面检索与混合检索：** 导入PDF或页面图像，用视觉向量或BM25与视觉结果的RRF融合返回相关页面。
- **双语RAG与Agent：** 固定RAG读取检索页面；Agent在搜索、读页、请求和时间限额内补查。中文问句先转为英文检索，按原问题语言回答。
- **证据与结构化输出：** 回答分列事实声明、已读页面引用和未解决信息，界面可跳转原页，CLI和浏览器保留JSON结果。
- **本地应用：** CLI与单模型常驻Gradio界面共用检索实现；页面编码分块保存，支持中断续接。
- **训练与评测：** 保留LoRA、查询蒸馏、BM25与神经检索对照，测量质量、延迟和资源开销。

## 快速开始

已验证Python 3.10、Ubuntu 22.04／WSL 2；GPU建库使用RTX 4060 Laptop 8GB。其他平台尚未实测。

### 42页CPU检索：已有本地候选时

将候选包的wheel与`demo-bundle.zip`放入同一目录，在该目录运行。首次加载需要获取配置固定的公开查询学生；CPU检索不需要教师权重、GPU或生成API。

```bash
python3.10 -m venv .venv
CPU_PY="$PWD/.venv/bin/python"
"$CPU_PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
"$CPU_PY" -m pip install 'foliorecall-0.1.0rc3-py3-none-any.whl[demo]'
"$CPU_PY" -m zipfile -e demo-bundle.zip demo-local
cd demo-local/demo-bundle
CUDA_VISIBLE_DEVICES= "$CPU_PY" -m foliorecall query --index . \
  --config configs/baseline.json --query-config configs/student-ml-cpu.json \
  'Which figure compares projections of the EU working-age population under baseline and no-migration scenarios?' --json
CUDA_VISIBLE_DEVICES= "$CPU_PY" -m foliorecall serve --index . \
  --config configs/baseline.json --query-config configs/student-ml-cpu.json --port 7864
```

打开`http://127.0.0.1:7864`。本地候选不等于公开下载；没有附件时可采用下面的源码和自有PDF流程。

### 源码安装与自有PDF

```bash
git clone https://github.com/linlin-is-me/FolioRecall.git
cd FolioRecall
REPO=$(pwd -P)
python3.10 -m venv .venv-cpu
CPU_PY="$REPO/.venv-cpu/bin/python"
"$CPU_PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
"$CPU_PY" -m pip install '.[demo,agent,eval]'
"$CPU_PY" -m foliorecall --help
```

自有PDF先在GPU环境导入和建库，再用CPU学生查询。[完整安装、PDF建库与复现步骤](doc/首版使用与评测.md#外部获取与复现)不依赖待发布附件。CPU查询能力不表示教师建库也已在CPU验证。

### RAG与Agent

问答使用生成API。准备页面库、匹配视觉索引，以及包含页面正文和来源的`public-corpus.json`；混合检索另需`hybrid.json`。环境变量`DASHSCOPE_API_KEY`和`DASHSCOPE_BASE_URL`由使用者在本机设置，不写入配置文件。

当前示例采用`qwen3.7-plus-2026-05-26`，关闭思考、温度0。已有索引可用`ask`或`serve --agent-config`入口，详见[问答命令与资料准备](doc/首版使用与评测.md#问答开发入口与当前验证边界)。本机演示复用256页JRC库；完整JRC材料、索引和凭据不随候选包提供。

## 架构


<p align="center">
  <a href="doc/assets/architecture-zh-CN-light.svg">
    <picture>
      <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="doc/assets/architecture-zh-CN-mobile-dark.svg">
      <source media="(max-width: 640px)" srcset="doc/assets/architecture-zh-CN-mobile-light.svg">
      <source media="(prefers-color-scheme: dark)" srcset="doc/assets/architecture-zh-CN-dark.svg">
      <img src="doc/assets/architecture-zh-CN-light.svg" width="1040" alt="离线导入页面，使用 Qwen3-VL 编码并保存 FAISS 索引与元数据，预览文件独立保存。在线使用 NanoVDR ML 编码查询，CPU FAISS 排序页面，CLI 或 Gradio 返回来源、物理页码和 JSON；Gradio 另行读取预览。">
    </picture>
  </a>
</p>

默认采用**原始 Qwen3-VL-Embedding-2B 建库，公开 NanoVDR ML 执行 GPU BF16 查询**。CPU FP32 查询可复用同一原始教师索引。LoRA 需要自身的匹配页面索引，向量维度相同不足以证明兼容。CLI 不传 `--query-config` 时仍使用教师查询。

本项目实现模型接入、索引兼容检查、可恢复数据处理、CLI 与界面检索及受控实验。骨干和公开查询学生来自上游项目。

## 结果

### 双语问答

固定40题、两种语言、三个条件共240项首次尝试，237项结构合法，全部完成助手评审。下表为必需事实覆盖均值，每个条件的实际分母为76；没有必需事实的题目不计入该指标。

| 条件 | 必需事实覆盖 | 已知问答费用估算（元） |
|---|---:|---:|
| Visual RAG | 0.702193 | 2.200664 |
| Hybrid RAG | 0.736988 | 2.198824 |
| Hybrid Agent | 0.722295 | 7.352474 |

Agent相对Hybrid RAG的共同配对覆盖差为−0.014693（76对），费用更高，收益随问题变化。40题的语言和条件配对不当作240个独立问题。费用列为已知问答估算，不包含翻译、探针或未知预留；完整四项评分、配对结果、失败与计费口径见[A5报告说明](doc/首版使用与评测.md#a5-模型迁移与新一轮验收)。评分来源为助手，`human=not_checked`；独立人工复核是可选后续。

rc3已有两项CLI、四项实际浏览器问答验证，覆盖中文转换、跨文档回答、引用跳页和不足回答。六项用例只验证使用流程，不构成新的独立质量评测。

### 检索、训练与查询成本


HR、Computer Science、Finance-EN、Industrial 和 Pharmaceuticals 五个固定英文任务共包含 **12,969 页、1,489 条查询**。各任务保留完整候选库和原始分级标签。下表按领域等权平均；神经模型使用 GPU BF16，BM25 在 CPU 上使用官方页面 markdown。

| 查询模型 / 方法 | nDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| 原始 Qwen3-VL 教师 | 0.537845 | 0.479970 | 0.580596 |
| LoRA750 | **0.573712** | **0.504133** | **0.612973** |
| 公开 NanoVDR EN | 0.566762 | 0.497533 | 0.605997 |
| 公开 NanoVDR ML · 默认 | 0.567706 | 0.502397 | 0.611766 |
| 继续蒸馏 · 第 94 步 | 0.437830 | 0.390437 | 0.484633 |
| 页面级 BM25 | 0.516565 | 0.455363 | 0.553117 |

- **本地 LoRA 微调：** 相对原始教师，nDCG@10 提高 **0.035867**，四个领域提高，**Finance-EN 下降 0.021927**；查询延迟增加。
- **公开 ML 查询学生：** 同卡 BF16 下，编码 P50 比原始教师快 **6.23–7.65 倍**，该收益来自 NanoVDR 上游模型。本项目继续蒸馏后的 nDCG@10 相对公开 ML **下降 0.129877**，第 94 步保留为无收益对照。

| GPU 查询模型 | 各领域完整本地请求 P50 |
|---|---:|
| 原始教师 | 30.66–32.37 ms |
| LoRA750 | 44.72–45.07 ms |
| 公开 ML | 4.51–6.61 ms |

测量使用 **RTX 4060 Laptop 8GB**、batch1、PyTorch/FAISS 线程 4/1、5 次预热，各配置在独立进程中按固定查询顺序运行 3 轮。完整请求从输入文本计时至 Top-10 JSON 就绪，不包含模型加载、界面准备和浏览器渲染。表中范围来自各领域的 P50，未混合不同领域计算延迟统计。CPU 作为独立部署方案报告，不计作同设备编码加速。

[完整结果与评测协议](doc/首版使用与评测.md#本轮实测结果)包含 CPU 对照、配对置信区间、逐领域计时、内存、建库成本及误例分析。

## 个人实现与上游来源

本项目实现文档导入和续接、完整模型与索引接入、视觉与BM25融合、有限Agent工具循环、引用与JSON校验、分项计时、费用记录，以及CLI、常驻界面和隔离安装交付。训练与蒸馏对照用于判断质量和资源取舍。

Qwen和NanoVDR提供模型与权重，Sentence Transformers提供编码和训练接口；FAISS、BM25和Gradio复用现成库。公开学生的编码加速来自上游模型，本项目贡献是接入、实测比较与应用集成。[项目介绍与面试准备](doc/项目介绍与面试准备.md)说明实现范围和实测依据。

## 模型与文档

| 资产 | 状态 | 入口 |
|---|---|---|
| 原始Qwen教师 | 上游可获取 | [固定配置](configs/baseline.json) |
| 公开NanoVDR学生 | 上游可获取 | [ML CPU](configs/student-ml-cpu.json) · [ML GPU](configs/student-ml-cuda.json) |
| rc3与42页示例 | 本地候选，附件未发布 | [使用与演示](doc/首版使用与评测.md#本地发布准备与求职展示) |
| LoRA750、蒸馏94步及实验附件 | 历史本地候选，附件未发布 | [获取与复现](doc/首版使用与评测.md#外部获取与复现) |

[实验记录](doc/experiments.csv) · [训练与蒸馏](doc/查询学生与蒸馏.md) · [开发计划](doc/多模态文档检索项目开发计划.md#当前开发重点)

## 局限

- 已完成自动检查与助手评审，仍有事实误读、遗漏和不足判断错误；结构合法及引用存在不直接证明事实得到支持。
- 中文问句检索英文页面，原生中文文档检索尚未验证。应用面向本地单用户，生成调用远程API。
- 结果来自固定任务和单个训练seed；上游训练重叠与文档级独立性未完整核实，HR中20条查询曾用于流程检查。
- LoRA收益随领域变化，继续蒸馏未改善检索；公开学生的上游教师指令与本项目不同。
- GPU教师建库与CPU学生查询的资源需求不同；重排序、ONNX/int8和多用户部署尚未实现。

## 参与与许可

可在[Issues](https://github.com/linlin-is-me/FolioRecall/issues)提交命令、环境和可复现问题。开发约定见[AGENTS.md](AGENTS.md)。

自有代码采用[Apache-2.0](LICENSE)，上游来源见[NOTICE](NOTICE)。模型和数据遵循各自许可；示例保留European Union来源及素材许可，剔除已知第三方图片，不附完整PDF。代码许可不覆盖上游文档。
