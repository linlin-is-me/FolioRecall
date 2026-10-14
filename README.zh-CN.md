<p><img src="foliorecall/assets/wordmark.png" width="420" alt="FolioRecall 页寻标识"></p>

# FolioRecall · 页寻

**检索文档证据、读取原页、执行补查并提供引用的文档Agent。**

[English](README.md) · 简体中文

FolioRecall结合视觉检索与BM25，处理含文字、表格和图表的文档。受限Agent决定何时继续搜索和读页，界面展示实际执行步骤，支持英文与中文追问。本地Gradio应用常驻CPU查询模型与索引。

> `0.1.0`最终本地候选已完成：多轮 Agent、ONNX FP32 推荐后端、界面与首次使用提示、项目Logo均已整合，等待审核。新版源码尚未推送，附件尚未发布。
> [候选与审核入口](doc/v0.1.0发布说明.md) · [完整快速开始](doc/v0.1.0快速开始.md)

## 一个真实文档任务

询问两份欧洲报告如何描述劳动年龄人口与就业变化，再用中文追问证据页和分析时间范围。界面实时显示搜索、读页和模型请求，随后呈现事实声明、原页引用与保存的JSON。

![真实Agent追问与执行步骤（排版调整前）](doc/assets/v010-agent-followup.png)

当前界面将输入与会话、答案与证据分区，支持窄屏单列；原页按需展开。上图保留真实 API 回答，来自排版调整前的版本。[当前界面与验证](doc/首版使用与评测.md#v010-final-candidate)

<details><summary>实际请求进度与不足回答</summary>

![Agent请求执行中](doc/assets/v010-agent-running.png)
![土星问题的不足回答](doc/assets/v010-insufficient.png)

</details>

[五分钟演示与实现讲解](doc/项目介绍与面试准备.md#五分钟演示) · [使用与验证记录](doc/首版使用与评测.md#v010-多轮文档agent与审核候选)

## Agent执行流程

```mermaid
flowchart TD
    Q[当前问题与最近成功问答] --> C{是否为追问}
    C -->|是| W[一次请求改写为独立英文检索问句]
    C -->|否| T[按需转换中文问句]
    W --> M[模型选择下一动作]
    T --> M
    M --> S[search_pages：视觉与BM25融合检索]
    M --> P[read_page：页面图像与正文]
    S --> F[工具结果与剩余资源]
    P --> F
    F --> M
    M --> A[事实声明、本轮引用与缺失信息]
    F -->|资源到限| A
    A --> J[校验并保存JSON，展示原页]
```

固定RAG读取检索前六页。Agent最多执行三次搜索、读取六个不同页面、发起八次模型请求，输出预算为8000 tokens，任务活动时间上限180秒。进度来自执行器实际回调，不展示模型隐藏思考。

## 核心实现与个人贡献

- **检索装配：** 文档导入、分块恢复编码、FAISS索引、正文与页面绑定，以及视觉和BM25各Top-50的RRF融合，固定 `k=60`。
- **工具与证据管理：** 搜索和读页反馈、资源限额、本轮引用权限、结构化答案，以及SQLite调用费用记录。
- **多轮应用：** 最近三轮成功问答提供上下文；一次请求完成指代改写与中文转换，占用当前轮原有资源。历史答案不扩充引用权限。
- **本地执行：** 工作线程和队列传递实时步骤，共用请求锁串行访问检索模型。浏览器断开不重发任务，后台仍保存终态。
- **实测与交付：** CLI、常驻浏览器界面、隔离安装、便携示例、完整 ONNX 查询后端，以及质量和费用的配对对照；保留失败及训练无收益结果。

Qwen与NanoVDR提供上游模型和权重；Sentence Transformers、FAISS、BM25与Gradio提供现成基础设施。本项目贡献是集成、工具执行、应用交付和受控比较，公开学生的加速收益归于上游模型。[来源说明](NOTICE) · [面试材料](doc/项目介绍与面试准备.md)

技术栈为Python、PyTorch／Sentence Transformers、ONNX Runtime、FAISS／BM25、HTTPX、SQLite和Gradio。Agent采用Function Calling与自实现Python工具循环；当前未使用LangChain或LangGraph。[技术栈与代码入口](doc/首版使用与评测.md#current-stack)

## 快速体验

已验证 Python 3.10、Ubuntu 22.04／WSL 2。取得本地候选 wheel、`demo-bundle.zip` 和 `query-onnx-fp32.zip`。42 页示例包含向量、预览、正文和 BM25；独立模型附件提供完整查询学生，无需 GPU、教师权重或重新建库。

在附件目录执行，已有相同依赖环境可复用：

```bash
python3.10 -m venv .venv
CPU_PY="$PWD/.venv/bin/python"
"$CPU_PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
"$CPU_PY" -m pip install 'foliorecall-0.1.0-py3-none-any.whl[onnx,demo,agent]'
"$CPU_PY" -m zipfile -e demo-bundle.zip demo-local
"$CPU_PY" -m zipfile -e query-onnx-fp32.zip demo-local
cd demo-local/demo-bundle
CUDA_VISIBLE_DEVICES= "$CPU_PY" -m foliorecall query --index . \
  --config configs/baseline.json --query-config configs/student-ml-cpu-default.json \
  --retrieval-config hybrid.json \
  'Which figure compares projections of the EU working-age population under baseline and no-migration scenarios?' --json
```

检索不需要生成 API。免费浏览器：

```bash
CUDA_VISIBLE_DEVICES= "$CPU_PY" -m foliorecall serve --index . \
  --config configs/baseline.json --query-config configs/student-ml-cpu-default.json \
  --retrieval-config hybrid.json --port 7867
```

停止上述服务，设置北京百炼环境变量后启动 RAG／Agent：

```bash
export DASHSCOPE_API_KEY='replace-with-your-key'
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RUN_DIR="../outputs/chat-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES= "$CPU_PY" -m foliorecall serve --index . \
  --config configs/baseline.json --query-config configs/student-ml-cpu-default.json \
  --retrieval-config hybrid.json --agent-config configs/agent-demo.json \
  --public-corpus public-corpus.json --output "$RUN_DIR" --port 7867
```

打开 http://127.0.0.1:7867。启用生成服务时默认进入文档问答，否则显示页面检索。顶部列出当前文档库；点击首问示例填入问题与模式，再提交，示例不会自动调用模型。追问示例与新会话问题单独标注。来源显示简称与物理页码，完整文件名保留在提示、原页详情及 JSON 中。新建会话清空上下文，JSON 与终态记录保存在本地。

问答需自行配置密钥及实际服务余额，示例 20 元为支出上限。切换会话不清空费用；重复启动使用新的记录目录。模型固定 `qwen3.7-plus-2026-05-26`，关闭思考。

[完整快速开始](doc/v0.1.0快速开始.md)包含同版本旧 wheel 替换、int8 可选切换和 PyTorch 回退。旧配置省略后端字段时继续使用 PyTorch；产品 CLI、API 与答案结构兼容。完整 JRC 库、完整 PDF 和凭据不随候选提供，查询模型作为独立附件分发，不进入源码或 wheel。[使用说明](doc/首版使用与评测.md)另含源码安装与自有 PDF 建库入口。

## 质量、费用与误例

**原单轮A5版本**完成40题、双语、三条件共240项首次终态与助手评审，237项结构合法。结构合法率不等于回答准确率；必需事实覆盖每条件有76项适用结果。

| 条件 | 必需事实覆盖 | 已知问答费用估算（元） |
|---|---:|---:|
| Visual RAG | 0.702193 | 2.200664 |
| Hybrid RAG | 0.736988 | 2.198824 |
| Hybrid Agent | 0.722295 | 7.352474 |

Agent对Hybrid RAG的覆盖配对差为−0.014693，共76对，费用更高，收益随问题变化。费用不含转换、探针和未知预留。语言和条件变体不构成240个独立问题。评分来自助手，保持 `human=not_checked`；独立人工复核为可选后续。[完整分母、失败与费用](doc/首版使用与评测.md#a5-模型迁移与新一轮验收)

v0.1.0多轮界面另有功能验证，A5不作为新路径的质量基准。已有误例包括遗漏适用条件、证据范围不足却扩大结论，以及不足判断错误；引用ID合法仍需核对语义支持。

五轮真实浏览器问答通过进度、追问、重置、引用和JSON检查，共25次API请求，已知费用估算0.455042元。跨文档用例只读取人口报告，遗漏就业报告证据，追问延续了遗漏；截图保留这项局限。土星问题返回空事实声明与不足判断。这些用例验证使用流程，不代表整体回答准确率。

另做12轮×基线／候选的Agent证据覆盖开发对照，24项均有助手评审。候选降低请求与费用，但必需事实覆盖0.6402→0.6061，跨文档核心遗漏仍在，未采用；默认行为与原候选包保持不变。[开发结果与局限](doc/首版使用与评测.md#agent证据覆盖开发对照2026-10-14)

## 检索架构与实验

![离线页面教师与在线查询学生](doc/assets/architecture-zh-CN-light.svg)

原始Qwen3-VL-Embedding-2B离线编码页面，公开NanoVDR ML在线编码查询。CPU FP32查询复用匹配的原教师索引；LoRA需要另建匹配索引。

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

### 新增CPU后端对照

同一 CPU、200 条冻结开发查询／1000 页，batch1、编码 4 线程，三轮交替测量。原入口计划 FAISS 单线程，但测量元数据记录实际为 4 线程；下表保留该轮实测值：

| 后端 | 完整请求P50 | 预热RSS | 开发集nDCG@10 |
|---|---:|---:|---:|
| PyTorch FP32 | 19.33 ms | 636.0 MiB | 0.972699 |
| ONNX FP32 | 8.62 ms | 817.0 MiB | 0.972699 |
| ONNX动态int8 | 4.79 ms | 581.1 MiB | 0.969008 |

ONNX FP32在五领域1489题的检索指标与PyTorch一致，约2.24倍提速，但内存更高，选作最终本地候选的 CPU 默认。int8约4.04倍提速、包体积减少72.2%，五域等权nDCG下降0.005031，保留为可选配置。只替换查询执行，包含池化、投影和归一化；时间不含加载、显示与生成，不代表问答同比加速。旧 PyTorch 体验入口继续兼容。该轮历史计时不与后来搜索阶段单线程的多深度实验作严格同条件比较。[完整领域结果与导出命令](doc/首版使用与评测.md#onnx-cpu-backends) · [安装态真实界面](doc/assets/onnx-fp32-retrieval.png)

另完成 **LLaVE教师训练对照**：同256条训练查询、64批，开发集nDCG@10为0.978928，普通CachedMNRL为0.977083，增加0.001845，Recall保持0.995。200题中仅2题改善、1题退化，收益较小且集中，暂不承担新教师、匹配学生与索引迁移成本，保留实验结果。[详细对照与局限](doc/首版使用与评测.md#llave-training)

多深度学生已完成实际四层裁剪、独立训练与四／六层联合训练的六组对照。联合四层参数减少20.7%，但开发集nDCG@10为0.495222，低于公开六层0.972699，故保留现有六层ONNX部署。[对照与负结果](doc/首版使用与评测.md#multidepth-query)。

## 能力边界

- 文档库为英文，中文问题和追问检索英文页面；原生中文文档检索尚未验证。
- 本地单用户应用，远程API生成；会话保存在浏览器状态，刷新清空，无持久会话数据库或多用户部署。
- 可能误读图表、遗漏事实或引用不足证据；历史回答用于理解指代，不是已核验事实。
- 固定任务、单个训练seed和助手评分限制泛化结论，上游训练重叠未完全确认。
- 检索无生成费用，问答需要密钥和服务余额；未实现重排序或任意外部工具。

## 文档与许可

[使用与评测](doc/首版使用与评测.md) · [实验CSV](doc/experiments.csv) · [开发计划](doc/多模态文档检索项目开发计划.md#当前开发重点) · [开发约定](AGENTS.md)

代码采用[Apache-2.0](LICENSE)，上游模型与资料保留各自许可。示例保留欧盟署名、排除已确认的第三方图像，不含完整PDF；[NOTICE](NOTICE)记录来源。可在[Issues](https://github.com/linlin-is-me/FolioRecall/issues)提供命令、环境和可复现问题。
