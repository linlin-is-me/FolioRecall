# FolioRecall（页寻）

Efficient Visual Document Retrieval

项目仓库：[linlin-is-me/FolioRecall](https://github.com/linlin-is-me/FolioRecall)。

FolioRecall 面向英文视觉文档检索，计划支持 PDF / 页面图像导入、页面编码和索引、检索微调、查询蒸馏及本地演示。离线由多模态教师编码页面，在线使用教师或与其对齐的查询学生搜索页面。

## 当前状态

2026-09-09：第一阶段的命令行检索、原始模型评测和小规模训练验证已跑通。两份 HR 原始 PDF 共 45 页完成编码、索引保存加载与查询，固定图表查询 Top-1 命中正确文档和物理第 10 页。HR 完整 1110 页上的 20 条固定英文查询已评分；VDR 的 8 条训练、4 条开发查询完成 4 步 LoRA 训练、适配器重载与独立索引检索。当前结果证明流程可运行，尚无正式训练收益或多领域评测结论；界面未接入。

当前任务、阻塞与下一步统一见[开发计划的当前开发重点](doc/多模态文档检索项目开发计划.md#当前开发重点)。

2026-09-10：阶段一收尾完成。HR 清单与索引元数据已补齐从 0 开始、按官方 Parquet 行顺序记录的 `row_index`；它与页面 ID、物理页码分别保存。评测在加载模型前校验 `--data/pages.jsonl` 与索引的完整页面 ID 集合，拒绝缺页、多页、重复 ID 或空候选库。第二阶段接续状态见下节。

## 第二阶段接续

**最新进展：** 2026-09-11 750/750 步已完成，续训段耗时 10078.0 秒（约 2.80 小时），224 个 LoRA 张量更新，最终 checkpoint-750 及优化器、调度器、随机状态已保存，训练进程已退出。按最新授权已启动独立开发建库和评测，入口记录 `outputs/stage2/lora/final-evaluation-launch.json`、日志 `final-evaluation.log`、状态 `final-evaluation-status.json`；候选索引 `outputs/stage2/lora/checkpoint-750/dev-index`，评测输出 `checkpoint-750/dev-evaluation/result.json`。目前评测运行中，尚无最终质量结论，每30分钟监控继续接续，完成后比较原始、339步和最终结果。

**2026-09-11 最新执行范围：** 用户随后授权：完成 750 步并保存退出后，由每 30 分钟监控自动接续冻结开发集的完整建库、评测和质量分析，比较原始模型、第 339 步与最终结果；运行中的 checkpoint-only 保持不变，评测独立执行。不开启额外训练、外部测试或蒸馏；若预算停止时未达 750 步，先报告等待，不自动延长。

2026-09-11 12:18（北京时间）按用户授权从 `checkpoint-339` 续跑 5 小时，使用 `--max-seconds 18000 --checkpoint-only`。目标仍为 750 步，配置与 seed 42 不变；本段保存检查点时跳过开发评测，达到最终目标也跳过编码重载探测，仅保留训练检查和可恢复状态。新增开关属于运行选项，记入 run.json，不改变恢复匹配的模型/组批配置。7 项相关 CPU 测试通过，实际恢复记录 `outputs/stage2/lora/resume-state-339.json` 确认步数 339、224 组优化器状态及确定性模式。运行代码 `b5cc725`，日志 `outputs/stage2/lora-resume-339.log`，PID/命令记录 `outputs/stage2/lora-resume-339-launch.json`；每 30 分钟监控已恢复。本段运行中，结束后只核实和记录检查点，不分析或评测，不自动增加预算。计时在训练开始后按完整步骤检查，加载及保存可额外耗时。


2026-09-10：阶段二数据准备与普通训练入口已完成，32 条查询的资源短跑及断点恢复通过。主训练、负例错误对照和教师选择尚未开展；资源预算确认后再启动主训练。

`data/vdr-stage2` 已完成 19 个英文源分片的逐片提取，固定 3,000 条训练查询、200 条开发查询、5,763 个训练页面和 1,000 个开发页面，共 6,763 页、约 1.48 GiB 图像，包含 1,445 个无查询页面。使用 seed 42，先划分全量页面池，再排除规范化重复查询组涉及的页面、抽样并固定上游顺序的有效同侧负例。开发任务位于 `data/vdr-stage2/dev`，所有候选共用其查询、标签和完整页面库。原始文档身份未核实，隔离仅达页面级；现有 20 条 HR 查询未发现规范化文本重合，不等于跨来源图像审计完成。

数据来源、revision、排除原因和划分见 `data/vdr-stage2/source.json`、`split.json`；逐片行号、解码检查和流量下界见 `extraction.json`。成功分片的获取与提取累计约 110 分钟，未包含所有失败重试、暂停和早期续传时间。英文源合计约 18.34 GiB，另有一个损坏分片重新获取，实际网络流量未完整计量。本次临时分片逐片删除，已有缓存保留。最终引用检查见 `outputs/stage2/data-check/final/result.json`。

前次已有 3 项数据测试和 3 项训练 CPU 测试通过。普通训练使用 SentenceTransformerTrainer、CachedMNRL 和语言侧共享 LoRA，配置集中在 [configs/lora-baseline.json](configs/lora-baseline.json)。支持跨批页面复用，禁止批内正负页面冲突；此前 3,000 条记录组成 750 个四样本批次，无尾批。预处理包装与原编码路径一致，恢复适配器时复用现有模型，优化器、调度器和随机状态由训练器接续。阶段一 `train-smoke` 保留。

历史资源短跑的数据划分和模型初始化使用 seed 42，组批器实际使用默认 seed 0；原始配置与数值保留并补记差异。后续 `train` 从配置读取 seed 42，显式绑定 sampler，统一 Python、NumPy 和 Torch 随机状态；`train.full_determinism=true` 复用 Transformers 的确定性模式，在模型加载前启用并交给 Trainer 保持。新运行保存 `batching_version=2` 和实际组批 seed，旧格式或不同生效配置不能直接恢复，不改写历史检查点。统一 seed 本身不保证确定性算子行为，设置依据见 [PyTorch 2.8 复现说明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)。

以下为本机历史运行命令；旧输出不作为新代码的续训入口：

```bash
source scripts/activate_env.sh
python scripts/prepare_data.py vdr-stage2
HF_HUB_OFFLINE=1 python -m foliorecall train --config configs/lora-baseline.json \
  --data data/vdr-stage2 --output outputs/stage2/profile --profile --stop-after 4 --max-seconds 900
HF_HUB_OFFLINE=1 python -m foliorecall train --config configs/lora-baseline.json \
  --data data/vdr-stage2 --output outputs/stage2/profile --profile \
  --resume outputs/stage2/profile/checkpoint-4 --max-seconds 900
```

资源样本覆盖不同页面尺寸，文字、表格和图表检查见 `outputs/stage2/page-spot-check.json`。8 步损失与梯度有限，224 个 LoRA 张量更新、冻结参数无梯度；第 4 步恢复了 224 组优化器状态和对应学习率，始终采用相同的 8 步调度目标。未启用梯度检查点，峰值分配显存约 7.15 GiB。两段训练合计 279.7 秒，包含组批与检查点保存；模型加载约 26–29 秒。适配器重载的两页向量最大差为 0.000228，容差为 0.001，配套索引检索通过。记录在 `outputs/stage2/profile/result-step-4.json`、`result.json`、`resume-state-4.json` 和 `probe-index`。该适配器只验证资源与流程，不参与教师选择。

训练输出分别记录有效查询展示次数、步骤计算时间、训练段时间和包含模型加载/导出重载的进程内时间。恢复时按检查点裁剪进度，旧记录留在当次 `run-from-*/previous-*.json`；后续检查点、索引和输出配置同步归档，避免重跑碰到已有索引或覆盖旧权重。`archived-artifacts.json` 保留原路径与归档位置；这些快照保留旧绝对路径供审计，不作为当前推理入口。训练显存取有效路径的逐步峰值，与检查点内 `dev-build.json`、`dev-result.json` 分开。普通训练启动前检查开发查询和标签文件，评测结束或异常时恢复训练状态。

本轮 9 项相关 CPU 测试通过，实际 sampler 与 32/3,000 查询的保存安排完全一致，分别为 8/750 个四样本批次。复用原有 8 条训练、4 条开发查询及 24 张图像补齐专用回归清单 `outputs/stage2/train-fixes-data`，完成普通训练两步、保存后开发评测再反传、旧检查点恢复、归档与重载。未启用确定性模式时，连续/恢复页面向量差曾达 0.006913，记录在 `outputs/stage2/train-fixes-recovery-check`，不记为数值复现通过。开启确定性模式后，连续与恢复运行的 224 个适配器张量逐位一致，两页索引向量差为 0；单次保存重载向量差 0.000239，低于 0.001 容差，详见 `outputs/stage2/deterministic-recovery-check/result.json`。这是本机小样本实测，不保证跨硬件或版本逐位一致，也不是阶段二主训练或质量基线。早期缺少开发标签、旧索引冲突的失败日志和产物均保留。

当前可运行的普通训练与恢复命令如下；输出目录已有完成结果，复跑时选用新目录：

```bash
HF_HUB_OFFLINE=1 python -m foliorecall train --config configs/lora-baseline.json \
  --data outputs/stage2/train-fixes-data --output outputs/stage2/train-deterministic --max-seconds 600
HF_HUB_OFFLINE=1 python -m foliorecall train --config configs/lora-baseline.json \
  --data outputs/stage2/train-fixes-data --output outputs/stage2/train-deterministic \
  --resume outputs/stage2/train-deterministic/checkpoint-1 --max-seconds 600
```

训练段时间覆盖当次调用的组批、保存和开发评测；有效步历史与实际尝试耗时分别解读，不采用框架对中断任务汇总的 samples/sec。

原始模型的固定内部开发任务也已跑通：

```bash
HF_HUB_OFFLINE=1 python -m foliorecall index \
  --pages data/vdr-stage2/dev/pages.jsonl --output indexes/vdr-dev-original
HF_HUB_OFFLINE=1 python -m foliorecall evaluate --data data/vdr-stage2/dev \
  --index indexes/vdr-dev-original --output outputs/stage2/dev-original
```

采用 `configs/baseline.json`，完整 1,000 页建库 497.7 秒、峰值分配显存约 4.24 GiB；200 条查询的 nDCG@10 为 0.9692、Recall@5/10 均为 0.995，驻留模型查询 P50/P95 为 30.9/39.1 ms。187 条查询 Top-1 命中原标正页，1 条未在 Top-10 召回；逐查询排名见 `outputs/stage2/dev-original/result.json`。原始模型在这组内部任务上已接近上限，区分方案的能力有限；尚无微调收益、稳定性或外部泛化结论，不修改冻结任务来追求差异。

资源估算与具体未召回查询见 `outputs/stage2/resource-summary.json`，运行版本与配置入口见 [实验摘要](doc/experiments.csv)。旧非确定性短跑曾外推 750 步约 7.3 小时，含评测建议 8–10 小时；这不是新确定性配置的预算承诺。确定性模式的本机校准现已完成，最新估算见下文；不同页面组合、功耗和温度会影响实际耗时，完整 LoRA 开发建库成本尚未实测。此前规定的短跑、恢复及原始模型评测已在一小时计算预算内完成，无需为用满预算重复运行。

2026-09-10 确定性资源校准已完成：代码 `d6e59d7`、运行时工作区干净，使用同一组 32 条查询，seed 42、`batching_version=2`、确定性算法开启。连续 8 步完成 32 次查询展示，自动保存第 4、8 步，未启用梯度检查点。224 个 LoRA 张量更新，损失和梯度有限，冻结范围检查通过；重载向量最大差 0.000318，配套探测索引检索通过。本轮未修改实现，复用此前 9 项 CPU 测试及确定性恢复验证。实际命令如下，已有完整结果，勿覆盖重跑：

```bash
source scripts/activate_env.sh
HF_HUB_OFFLINE=1 python -u -m foliorecall train \
  --config configs/lora-baseline.json --data data/vdr-stage2 \
  --output outputs/stage2/profile-seed42-deterministic \
  --profile --max-seconds 1800 \
  > outputs/stage2/profile-seed42-deterministic.log 2>&1
```

外接电源、平衡电源方案下，每步 24.18–28.20 秒，步骤计算合计 209.16 秒，训练段 213.85 秒，初次模型加载 28.44 秒，完整调用 285.87 秒；剩余 43.58 秒为其他准备、导出与重载等开销。训练峰值分配显存 7.13 GiB；总显存减该值约 886 MiB，还需容纳显示、驱动及分配器预留，不能视为可用余量。未出现 OOM，但未实测所有主训练批次及长时间温度变化。

按 `213.85 / 8 * 750` 外推训练约 5.57 小时；加初次加载、两次原始开发建库参考共 995.32 秒及查询参考 15.64 秒，合计约 5.86 小时，增加 25% 余量后约 7.32 小时，向上取整建议本机预算 **8 小时**。查询成本使用既有驻留 P95 乘查询数作为参考；确定性 LoRA 的完整开发建库尚未实测，短跑保存开销也按步数外推。该估算仅适用于当前设备和条件，不能据它与历史 7.3 小时的差异声称确定性模式提速。详情和运行提交、配置、样本、日志入口见 [新资源摘要](outputs/stage2/resource-summary-seed42-deterministic.json)，旧摘要保留。

[页面抽查记录](outputs/stage2/page-spot-check.json) 的 `retrieval_error_review` 已检查全部 13 条非 Top-1 查询的原标页及 Top-1 页。助手视觉判断暂分为：6 条具体证据排序问题，5 条可能存在其他相关页，1 条查询指代含糊，1 条证据不足。例如 NTP 时间、湖泊日期与 D0 公式有具体区分依据；屋顶农业挑战、零工优势及缓存访问可能涉及多页相关。唯一未召回的漫画角色查询缺少作品指代，原标图也没有身份文字。每例保留 ID、排名、两页路径、证据及不确定性；这些判断不是人工真值，没有改动标签，也不据此过滤训练负例。当前证据不足以认定冻结开发任务不可用，保留 200 查询、1,000 页任务；若后续选模受这些争议影响，再作人工复核并在统一标签上重算候选。

2026-09-11 本机首段训练已按预算停止并完成开发评测，进程已退出，监控暂停。有效进度 339/750 步（45.2%）、1,356 次查询展示；224 个 LoRA 张量更新、冻结参数无梯度，未记录非有限损失或梯度错误。检查点 `outputs/stage2/lora/checkpoint-339` 保存适配器、优化器、调度器和随机状态；匹配编码配置 `encoding.json`、完整开发索引 `dev-index`、建库记录 `dev-build.json`、指标与排名 `dev-result.json` 均在检查点内。段结果为 `outputs/stage2/lora/result-step-339.json`，`complete=false`；未达到最终导出与重载探测分支，因此没有 `result.json`，本检查点尚未实际恢复验证，既有小样本恢复验证继续有效。

同一冻结开发任务下，原始模型与 339 步模型的 nDCG@10 为 0.969236 / 0.978095，Recall@5 为 0.995 / 0.990，Recall@10 为 0.995 / 0.995。排序指标小幅上升，Top-5 召回下降；仅为中途内部开发结果，不代表完整基线或稳定泛化收益，不选择教师。此次 1,000 页开发建库实耗 4,597.99 秒（76.63 分钟），明显高于原始参考；训练段统计也包含该评测，不与优化步骤耗时混用。短跑预算估算未能代表主训练实际速度，具体原因尚未核实。

下一步先结合中途结果与资源异常确定续训设备和新增预算，再从 checkpoint-339 接续剩余 411 步；保持原配置、目标 750 步及输出目录，通过 `--resume` 指定检查点。未经新预算确认不续训，不启动错误对照、教师选择或蒸馏。

以下为已结束运行的启动记录：

2026-09-11 00:19（北京时间），用户已授权在本机按 8 小时预算启动 3,000 查询普通 LoRA 基线。后台进程使用 `configs/lora-baseline.json`，从原始模型初始化，seed 42、确定性模式，目标 750 步，第 375、750 步保存并开发评测；`--max-seconds 28800` 在完整步骤后检查，退出保存及评测可能额外耗时。已确认首个优化步骤完成，尚无主训练质量结果。启动记录 `outputs/stage2/lora-launch.json`，日志 `outputs/stage2/lora.log`，配置、实际组批、进度及检查点在 `outputs/stage2/lora`。Codex 定时监控 `foliorecall` 每 30 分钟检查，完成或异常时报告，不自动延长预算。应用需保持运行以执行监控；训练进程独立运行于 WSL，勿关机或关闭 WSL。短跑适配器不参与教师选择，教师与蒸馏尚未启动。

```bash
source scripts/activate_env.sh
HF_HUB_OFFLINE=1 python -u -m foliorecall train \
  --config configs/lora-baseline.json --data data/vdr-stage2 \
  --output outputs/stage2/lora --max-seconds 28800
```

上述启动命令仅供历史记录，本段任务已结束，不要从头重复启动。后续读取半程与最终开发结果，再决定错误对照；失败或预算结束时先核实已保存状态，不自动重启。

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
