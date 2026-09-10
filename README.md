# FolioRecall（页寻）

Efficient Visual Document Retrieval

项目仓库：[linlin-is-me/FolioRecall](https://github.com/linlin-is-me/FolioRecall)。

FolioRecall 面向英文视觉文档检索，计划支持 PDF / 页面图像导入、页面编码和索引、检索微调、查询蒸馏及本地演示。离线由多模态教师编码页面，在线使用教师或与其对齐的查询学生搜索页面。

## 当前状态

2026-09-09：第一阶段的命令行检索、原始模型评测和小规模训练验证已跑通。两份 HR 原始 PDF 共 45 页完成编码、索引保存加载与查询，固定图表查询 Top-1 命中正确文档和物理第 10 页。HR 完整 1110 页上的 20 条固定英文查询已评分；VDR 的 8 条训练、4 条开发查询完成 4 步 LoRA 训练、适配器重载与独立索引检索。当前结果证明流程可运行，尚无正式训练收益或多领域评测结论；界面未接入。

当前任务、阻塞与下一步统一见[开发计划的当前开发重点](doc/多模态文档检索项目开发计划.md#当前开发重点)。

2026-09-10：阶段一收尾完成，第二阶段尚未启动。HR 清单与索引元数据已补齐从 0 开始、按官方 Parquet 行顺序记录的 `row_index`；它与页面 ID、物理页码分别保存。评测在加载模型前校验 `--data/pages.jsonl` 与索引的完整页面 ID 集合，拒绝缺页、多页、重复 ID 或空候选库。

## 第二阶段接续

2026-09-10：已开始数据准备与普通训练入口开发，尚未运行主训练或选定教师。先执行 `python scripts/prepare_data.py vdr-stage2 --manifest-only` 生成固定清单，再执行 `python scripts/prepare_data.py vdr-stage2` 逐片提取原图。产物位于 `data/vdr-stage2`，开发任务位于其 `dev` 子目录；原有 `data/vdr` 保留为第一阶段验证数据。

固定 seed 42，先按全部页面 ID 划分 80%/20%，排除规范化后重复查询组涉及的页面，再抽取 3,000 条训练、200 条开发查询。每条训练查询保留一个上游顺序的有效同侧负例；开发候选库固定 1,000 页。仅生成清单的实际结果为 5,763 个训练页面、1,000 个开发页面，共 6,763 页。原始文档身份仍未核实，页面级隔离的局限保留。完整英文源读取预算约 18.34 GiB，逐片提取后删除本次临时分片，不清理已有缓存。

新增数据划分测试 `python -m unittest discover -s tests -p test_stage2_data.py -v` 已通过。实际图片提取、训练入口和资源短跑尚待完成；计划先做约一小时本机短跑，再由用户确定主训练设备与预算。

## 本机开发环境

| 项目 | 实际位置或配置 |
|---|---|
| 开发系统 | WSL 2，Ubuntu-22.04，Ubuntu 22.04.5 LTS；当前默认用户 root |
| Windows 工作副本 | `D:\myproject\FolioRecall` |
| WSL 工作副本 | `/mnt/d/myproject/FolioRecall`，与 Windows 共用代码 |
| 独立 venv | `/root/.venvs/foliorecall_env`，不继承其他环境的 site-packages |
| Python | 系统 Python 3.10.12 创建的 venv |
| PyTorch / torchvision | `2.8.0+cu126` / `0.23.0+cu126`，CUDA runtime 12.6 |
| Sentence Transformers / Transformers | `6.0.1` / `5.16.1`；其他已验证直接依赖见 requirements.txt |
| 新增训练与数据组件 | PEFT `0.20.0`、datasets `5.0.1`、pyarrow `25.0.1`；安装后 pip check 与新增导入通过，fsspec 调整为 `2026.6.0` |
| 模型与数据集缓存 | `HF_HOME=/mnt/d/foliorecall_cache/huggingface` |
| 安装临时目录 | `TMPDIR=/mnt/d/foliorecall_cache/tmp` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB；驱动 610.74 |

环境创建前确认 Ubuntu 的 VHDX 位于 C 盘。宿主机 C 盘约余 11.1 GiB、D 盘约余 25.0 GiB；WSL 内部 `df` 显示的虚拟容量不能代替宿主机余量。环境使用 Linux 文件系统，较大的模型缓存和安装临时文件使用 D 盘。已有 Windows、Conda 与其他 venv 保持原状。

安装后环境约占 6.0 GiB，宿主机 C 盘约余 6.53 GiB、D 盘约余 24.98 GiB；安装临时文件已自动释放。后续模型与数据继续放在 D 盘缓存目录，下载前按实际需要确认余量。

后续会话在 PowerShell 执行：

```powershell
wsl -d Ubuntu-22.04
```

进入 WSL 后：

```bash
cd /mnt/d/myproject/FolioRecall
source scripts/activate_env.sh
```

激活脚本只加载现有环境并设置缓存路径，不安装依赖、不下载模型、不运行检查。它记录本机实际路径；其他机器按自己的用户与磁盘位置调整。

## 首次安装与必要检查

本机环境建立后直接复用。以下是重新准备环境时的步骤，执行前先确认目标环境是否已经存在，以及实际磁盘余量。

```bash
python3.10 -m venv /root/.venvs/foliorecall_env
mkdir -p /mnt/d/foliorecall_cache/huggingface /mnt/d/foliorecall_cache/tmp
cd /mnt/d/myproject/FolioRecall
source scripts/activate_env.sh
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements.txt
```

本机安装使用已有 `uv 0.12.7` 的 `uv pip install --python /root/.venvs/foliorecall_env/bin/python --no-cache ...` 并发获取依赖；它写入同一 venv，不创建第二套环境。D 盘临时目录与环境分属不同文件系统，安装采用复制；使用 uv 时可加 `--link-mode=copy`。上面的 pip 命令提供同版本安装入口。

PyTorch 与 torchvision 采用[官方配对版本及 CUDA 12.6 wheel](https://pytorch.org/get-started/previous-versions/#v280)。[Qwen3-VL-Embedding 模型卡](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#usage)提供 PyTorch 2.8 和 Sentence Transformers 接入方式。当前依赖见 [requirements.txt](requirements.txt)，未安装 FlashAttention 或独立 CUDA Toolkit；新增训练组件的实际版本见上表。

首次接入、相关依赖变化或异常时才执行：

```bash
python -m pip check
python scripts/check_env.py
```

检查包括解释器隔离、必要依赖与 Qwen3-VL 类导入、合成空白页渲染、合成 FAISS 精确搜索、CPU 小矩阵运算，以及 GPU 可用时的一次 BF16 小矩阵运算。无需联网或模型权重；GPU 不可用时明确跳过 GPU 检查，仍可开展 CPU 工作。该检查不验证真实 PDF 导入、模型加载、LoRA 反传或检索质量。

### 本机验证记录

2026-09-09，在上述 WSL venv 中执行 `python -m pip check` 和 `python scripts/check_env.py`：

| 检查 | 结果 |
|---|---|
| 环境隔离 | `include-system-site-packages=false`，解释器位于上述 venv |
| 依赖一致性 | `No broken requirements found.` |
| 必要依赖与模型类导入 | requirements.txt 中的依赖、AutoProcessor 和 Qwen3VLModel 均成功导入；未加载模型权重 |
| PDF 渲染库 | 内存中新建 32×32 空白页面并渲染，尺寸检查通过 |
| FAISS | 3 个合成向量的内积精确搜索，返回 ID 与分数正确 |
| PyTorch CPU | 32×32 矩阵乘法结果检查通过 |
| RTX 4060 CUDA | CUDA 可用，32×32 BF16 矩阵乘法及同步后结果检查通过 |

环境稳定且依赖未改变时，直接激活并继续开发，无需重复以上检查。应用功能的实际验证状态见下节；查询学生的 CPU 快速体验尚未实现。

## 第一阶段命令

在仓库根目录执行 `source scripts/activate_env.sh`。以下命令只使用同一个环境；新增的 `peft`、`datasets` 按 requirements.txt 安装，已有未变更组件不必重装。

已跑通的数据准备、真实 PDF 导入和 CPU 核心检查：

```bash
python scripts/prepare_data.py pdfs
python -m foliorecall import data/pdfs/*.pdf --output data/demo
python scripts/prepare_data.py hr
python scripts/prepare_data.py vdr
python scripts/prepare_data.py model
python -m unittest discover -s tests -v
```

两份报告的来源、revision、许可入口保存在 `data/pdfs/sources.json`；页面清单在 `data/demo/pages.jsonl`，预览在同目录的文档子目录。物理页码从 1 开始，与页面印刷页码可能不同。缓存链接保留原文件名；重复导入未变化的文件复用已有页面，输入变化按路径、大小、修改时间和 DPI 更新对应结果。每次 import 输出本次输入的完整清单，不自动追加其他文档。损坏或加密 PDF 的跳过原因写入 `import-errors.json`。

实际验收：两份报告分别 20、25 页，物理页码连续；混合导入一个加密测试文件时，记录密码错误并保留其余 45 页。3 项 CPU 测试覆盖索引重载与来源映射、配置错配、分级多正例指标及损坏输入。记录保存在本地 `outputs/cpu-validation`、`outputs/data-validation`、`outputs/vdr-validation`。[人工查询案例](examples/demo_queries.json)对应人口报告物理第 10 页的 Figure 3，官方 HR 页面 ID 为 1028，印刷页码为 7；两条导入路径已视觉核对。

已跑通原始模型建库与查询：

```bash
python -m foliorecall index --pages data/demo/pages.jsonl --output indexes/demo-original
python -m foliorecall query --index indexes/demo-original 'Which figure compares projections of the EU working-age population under baseline and no-migration scenarios?'
```

45 页建库约 41.4 秒，PyTorch 峰值显存约 4.24 GiB；时间包含页面读取、编码和索引保存，不含模型加载。结构化查询结果见本地 `outputs/demo-query.json`，来源核对见 `outputs/demo-query-check.json`。当前 Sentence Transformers 的图像参数分组为 `image`；修复后两页探测不再出现未知参数警告，实际网格与输入形状保存在 `indexes/probe-original-validated/encoding-probe.json`。模型首次下载曾停顿，使用 aria2 HTTP 续传恢复，现已接回标准 Hugging Face 缓存；aria2 仅用于下载故障恢复，不是应用运行依赖。

完整 HR 评分与训练入口：

```bash
python -m foliorecall index --pages data/hr/pages.jsonl --output indexes/hr-original
python -m foliorecall evaluate --index indexes/hr-original --output outputs/hr-original
python -m foliorecall train-smoke --output outputs/train-smoke
```

默认配置为 `configs/baseline.json`，可用 `--config` 指定配置，查询加 `--json` 输出结构化结果。图像入口使用 `import --image-manifest <JSONL> --output <目录>`；每行提供 `page_id`、`doc_id`、`source`、从 1 开始的 `page_number`、`preview`，预览相对清单目录解析。HR 基准直接使用官方图像，不使用演示 PDF 的重新渲染图像。

本机 4 步训练已通过：语言侧注意力 LoRA 共 3,211,264 个参数，224 个张量实际更新；每步损失与梯度有限，冻结参数无梯度。损失依次为 0.167308、0.036332、0.015513、0.028853，各步使用不同样本，不据此判断收敛。训练耗时 221.1 秒，PyTorch 峰值显存约 7.04 GiB，未启用梯度检查点。适配器保存重载后两页向量最大绝对差为 0，随后完成 8 个开发页面建库与 4 条开发查询检索。完整记录在 `outputs/train-smoke/result.json`，适配器在 `outputs/train-smoke/adapter`，仅约 6.45 MB；仍需原始模型权重。

微调模型查询使用配套配置和索引：

```bash
python -m foliorecall query --config outputs/train-smoke/config.json \
  --index outputs/train-smoke/index 'your English query' --json
```

该入口已在独立进程用开发查询验证，正常返回 5 条带页码与预览信息的 JSON 结果，记录在 `outputs/train-smoke/cli-query.json`。原始模型配置与微调索引不可混用。后续训练显存不足时，用新的输出目录加 `--gradient-checkpointing` 重试；失败记录保留在 `failure.json`。

VDR 按列读取了完整英文元数据：94,225 页、53,512 条非空查询，保留无查询页面记录。切片服务不覆盖整库，本次仅从前 1000 行中选择满足隔离要求的 8 条训练查询、4 条开发查询及原始负页面，共 24 张图像。全部图像可解码，正负页面跨划分无交叉，负例均在原始标注列表内。选择规则、原始负例和排除项保存在 `data/vdr`；这组前部小样本仅检查流程，页面级隔离不等于已经确认原始文档隔离。正式训练应按源 Parquet 获取页面，不能沿用切片服务限制来代表全量数据。

HR 使用完整 1110 页候选库与种子 42 抽出的 20 条英文查询，保留多页标签和相关性等级。实际建库 1031.0 秒，PyTorch 峰值显存 4.24 GiB；nDCG@10 为 0.5951，Recall@5/10 为 0.5890/0.6829。模型驻留并预热一次后，查询编码与 FAISS 搜索的 P50/P95 为 52.0/67.4 ms，不含进程启动和模型加载；评测峰值显存约 3.98 GiB。结果见 `outputs/hr-original/result.json`，索引在 `indexes/hr-original`。这些结果只作早期流程检查，未用于调参，不代表完整 ViDoRe 成绩。nDCG 使用线性等级增益，Recall 对等级大于 0 的相关页面计算。

收尾验证命令为 `python -m unittest discover -s tests -p test_closeout.py -v`，2 项针对性 CPU 测试通过，包括遗漏非相关候选时在模型加载前报错。既有 1110 页的行号从缓存 Parquet 的元数据列回填，原字段与索引向量保持不变，记录在 `outputs/stage1-closeout`，索引目录的 `metadata-update.json` 指向该记录。此次未读取图像、重新编码或重跑训练，既有质量与资源数字仍来自 2026-09-09 的运行。

索引保存生效配置与页面映射；关键运行在产物目录保存 `run.json`、命令参数、必要的未提交差异和结果，训练同时保存样本清单。[实验摘要](doc/experiments.csv)关联实际代码版本、配置和本地产物；HR 评测与训练运行于干净的 `7b1bf91`，演示建库的未提交代码差异已随运行保存。已存在的索引拒绝覆盖，重建时使用新目录。修改输入、预处理或适配器时使用新产物目录并重建匹配索引。下载数据、模型、生成预览及索引均不提交 Git。

## 使用范围与上游来源说明

首版先接入英文页面检索，中文、答案生成、重排序及候选方法按开发计划择需启用。GPU 建库和训练以实际验证记录为准；使用查询学生的 CPU 快速体验路径尚未实现。

编码与训练接入 [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 和 [Sentence Transformers](https://github.com/huggingface/sentence-transformers) 的现成接口，精确搜索使用 [FAISS](https://github.com/facebookresearch/faiss)。本项目实现导入元数据、命令行、数据筛选与划分、索引配置匹配及检查脚本；尚未取得训练收益证据。NanoVDR 的查询学生与效率评测适配留待后续阶段。

## 开发文档

- [协作约定](AGENTS.md)
- [项目开发计划](doc/多模态文档检索项目开发计划.md)
- [分阶段开发提示词](doc/FolioRecall_分阶段开发提示词.md)

当前沿用已有 `doc/` 目录。实验与代码按实际开发逐步增加，不预建空模块或复杂 CI。大模型、原始数据、缓存与生成索引不纳入 Git 提交。
