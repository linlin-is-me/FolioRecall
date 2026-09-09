# FolioRecall（页寻）

Efficient Visual Document Retrieval

项目仓库：[linlin-is-me/FolioRecall](https://github.com/linlin-is-me/FolioRecall)。

FolioRecall 面向英文视觉文档检索，计划支持 PDF / 页面图像导入、页面编码和索引、检索微调、查询蒸馏及本地演示。离线由多模态教师编码页面，在线使用教师或与其对齐的查询学生搜索页面。

## 当前状态

2026-09-09：已接入指定远程仓库，独立开发环境通过最小验证。PDF 导入、原始模型加载、页面编码、建库查询、模型训练与正式评测尚未完成。环境检查使用合成数据，不代表第一阶段基础系统已经完成。

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

PyTorch 与 torchvision 采用[官方配对版本及 CUDA 12.6 wheel](https://pytorch.org/get-started/previous-versions/#v280)。[Qwen3-VL-Embedding 模型卡](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#usage)提供 PyTorch 2.8 和 Sentence Transformers 接入方式。当前依赖见 [requirements.txt](requirements.txt)，没有安装 FlashAttention、独立 CUDA Toolkit 或下载模型权重；PEFT、datasets 等组件在接入训练时再按需安装。

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

环境稳定且依赖未改变时，直接激活并继续开发，无需重复以上检查。真实 PDF 导入、Qwen3-VL 权重加载、页面编码、检索、LoRA 反传与训练保存重载均留待后续验证；CPU 示例体验也尚未实现。

## 使用范围与上游来源

首版先支持英文页面检索，中文、答案生成、重排序及候选方法按开发计划择需启用。完整使用路径计划在 GPU 环境为自有 PDF 建库；CPU 快速体验路径计划使用可分发的示例索引与查询学生，两者目前均未实现。

计划复用 [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)、[Sentence Transformers](https://github.com/huggingface/sentence-transformers)、[NanoVDR](https://github.com/Ryenhails/NanoVDR) 和 [FAISS](https://github.com/facebookresearch/faiss)。本轮只完成环境准备，尚未复现上游训练或获得性能结果。后续引入代码和分发权重时保留相应来源、许可与适配说明。

## 开发文档

- [协作约定](AGENTS.md)
- [项目开发计划](doc/多模态文档检索项目开发计划.md)
- [分阶段开发提示词](doc/FolioRecall_分阶段开发提示词.md)

当前沿用已有 `doc/` 目录。实验与代码按实际开发逐步增加，不预建空模块或复杂 CI。大模型、原始数据、缓存与生成索引不纳入 Git 提交。
