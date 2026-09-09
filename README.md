# FolioRecall（页寻）

Efficient Visual Document Retrieval

项目仓库：[linlin-is-me/FolioRecall](https://github.com/linlin-is-me/FolioRecall)。

FolioRecall 面向英文视觉文档检索，计划支持 PDF / 页面图像导入、页面编码和索引、检索微调、查询蒸馏及本地演示。离线由多模态教师编码页面，在线使用教师或与其对齐的查询学生搜索页面。

## 当前状态

2026-09-09：两份 HR 原始 PDF 共 45 页已导入；HR 完整 1110 页、20 条固定英文查询，以及 VDR 的 8 条训练查询、4 条开发查询和 24 张图像已准备并通过数据检查。命令行与模型接口已接入，CPU 核心测试通过。原始模型下载仍在处理，GPU 编码、检索质量和训练尚未实测，第一阶段未完成。

当前任务、阻塞与下一步统一见[开发计划的当前开发重点](doc/多模态文档检索项目开发计划.md#当前开发重点)。

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
python -m unittest discover -s tests -v
```

两份报告的来源、revision、许可入口保存在 `data/pdfs/sources.json`；页面清单在 `data/demo/pages.jsonl`，预览在同目录的文档子目录。物理页码从 1 开始，与页面印刷页码可能不同。缓存链接保留原文件名；重复导入未变化的文件复用已有页面，输入变化按路径、大小、修改时间和 DPI 更新对应结果。每次 import 输出本次输入的完整清单，不自动追加其他文档。损坏或加密 PDF 的跳过原因写入 `import-errors.json`。

实际验收：两份报告分别 20、25 页，物理页码连续；混合导入一个加密测试文件时，记录密码错误并保留其余 45 页。3 项 CPU 测试覆盖索引重载与来源映射、配置错配、分级多正例指标及损坏输入。记录保存在本地 `outputs/cpu-validation`、`outputs/data-validation`、`outputs/vdr-validation`。[人工查询案例](examples/demo_queries.json)对应人口报告物理第 10 页的 Figure 3，官方 HR 页面 ID 为 1028，印刷页码为 7；两条导入路径已视觉核对。

下面为已接入、等待实际模型验证的命令，不表示已跑通：

```bash
python scripts/prepare_data.py model
python -m foliorecall index --pages data/demo/pages.jsonl --output indexes/demo-original
python -m foliorecall query --index indexes/demo-original 'How does migration affect the projected EU working-age population?'
python -m foliorecall index --pages data/hr/pages.jsonl --output indexes/hr-original
python -m foliorecall evaluate --index indexes/hr-original --output outputs/hr-original
python -m foliorecall train-smoke --output outputs/train-smoke
```

默认配置为 `configs/baseline.json`，可用 `--config` 指定配置，查询加 `--json` 输出结构化结果。图像入口使用 `import --image-manifest <JSONL> --output <目录>`；每行提供 `page_id`、`doc_id`、`source`、从 1 开始的 `page_number`、`preview`，预览相对清单目录解析。HR 基准直接使用官方图像，不使用演示 PDF 的重新渲染图像。

首次训练显存不足时，用新的输出目录加 `--gradient-checkpointing` 重试；仍失败则保留 `failure.json`，不将流程检查写成训练成功。训练产物只保存 LoRA，加载时仍需原始模型；微调索引与原始索引不可混用。

VDR 按列读取了完整英文元数据：94,225 页、53,512 条非空查询，保留无查询页面记录。切片服务不覆盖整库，本次仅从前 1000 行中选择满足隔离要求的 8 条训练查询、4 条开发查询及原始负页面，共 24 张图像。全部图像可解码，正负页面跨划分无交叉，负例均在原始标注列表内。选择规则、原始负例和排除项保存在 `data/vdr`；这组前部小样本仅检查流程，页面级隔离不等于已经确认原始文档隔离。正式训练应按源 Parquet 获取页面，不能沿用切片服务限制来代表全量数据。

HR 使用完整 1110 页候选库与种子 42 抽出的 20 条英文查询，只作早期流程检查；nDCG 使用线性等级增益，Recall 对等级大于 0 的相关页面计算。

索引保存生效配置与页面映射；关键运行在产物目录保存 `run.json`、命令参数、必要的未提交差异和结果，训练同时保存样本清单。已存在的索引拒绝覆盖，重建时使用新目录。修改适配器时也使用新产物目录并重建匹配索引，避免路径相同而权重变化。下载数据、模型、生成预览及索引均不提交 Git。

## 使用范围与上游来源说明

首版先接入英文页面检索，中文、答案生成、重排序及候选方法按开发计划择需启用。GPU 建库和训练以实际验证记录为准；使用查询学生的 CPU 快速体验路径尚未实现。

编码与训练接入 [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 和 [Sentence Transformers](https://github.com/huggingface/sentence-transformers) 的现成接口，精确搜索使用 [FAISS](https://github.com/facebookresearch/faiss)。本项目实现导入元数据、命令行、数据筛选与划分、索引配置匹配及检查脚本；尚未取得训练收益证据。NanoVDR 的查询学生与效率评测适配留待后续阶段。

## 开发文档

- [协作约定](AGENTS.md)
- [项目开发计划](doc/多模态文档检索项目开发计划.md)
- [分阶段开发提示词](doc/FolioRecall_分阶段开发提示词.md)

当前沿用已有 `doc/` 目录。实验与代码按实际开发逐步增加，不预建空模块或复杂 CI。大模型、原始数据、缓存与生成索引不纳入 Git 提交。
