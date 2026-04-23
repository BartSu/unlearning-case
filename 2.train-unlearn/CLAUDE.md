# CLAUDE.md —— 阶段 ②「Unlearn 训练」

本目录的唯一产物是 **每个 forget set 一个 unlearn checkpoint**（`unlearn/saves/...`），供阶段 ③ cross-PPL 评测消费。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)。数据契约见 [`../1.data-preparation/CLAUDE.md`](../1.data-preparation/CLAUDE.md)。本文件补充阶段 ② 专有的**代码来源、环境约束和下游契约**。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 用 OpenUnlearning 框架 unlearn Llama-3.1-8B-Instruct | 修改数据集（属阶段 ①） |
| 为每个 triplet 生成一份 checkpoint | 计算 cross-PPL（属阶段 ③） |
| 记录训练日志（`logs_wikitext_unlearn*`） | 自行升级框架依赖版本 |

## `open-unlearning/` —— 上游 vendored fork

**来源**：[locuslab/open-unlearning](https://github.com/locuslab/open-unlearning/tree/main) ·  License: MIT · 以 vendored 形式内嵌（非 git submodule，直接拷贝源码）。

**角色**：提供 `src/train.py` + Hydra 配置（`configs/`）+ Trainer 抽象（`src/trainer/unlearn/`）。GradAscent / NPO / GradDiff 等方法直接复用上游实现，不自行重写。

### 🚨 环境配置必须遵循上游

按上游 `open-unlearning/README.md` 的 Quickstart 部分走，**不要自己挑版本**：

```bash
conda create -n unlearning python=3.11
conda activate unlearning
cd 2.train-unlearn/open-unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3
```

关键版本钉死在 `open-unlearning/requirements.txt`（`torch==2.4.1`, `transformers==4.51.3`, `accelerate==0.34.2`, `deepspeed==0.15.4`, `hydra-core==1.3` 等）。

**硬性要求**：
- 升级任何依赖版本前，先确认上游最新 main 是否已升级，**优先同步上游**而不是本地 patch。
- 本地 Python 必须 ≥ 3.11（上游 `python_requires=">=3.11"`）。
- Flash-attn 不在 `requirements.txt` 里，必须手动装 `2.6.3`（上游 Quickstart 指定），否则 `attn_implementation=sdpa` 虽可退回但训练速度会下降。

### vendored fork 的维护规则

- 优先**不改** `open-unlearning/src/` 下的核心代码（`train.py`, `trainer/`, `data/`, `evals/`）—— 要改先问：能否通过 Hydra config override / 新 Trainer 子类 / 外层 wrapper 脚本达成同样目的？改上游代码会失去和 upstream `git diff` 同步的能力。
- 本仓库对 `open-unlearning/` 的任何本地改动，都应在 commit message 或 PROGRESS.md 里标注"local patch to upstream"，方便以后合并上游更新。
- `open-unlearning/saves/` 和 `open-unlearning/logs/` 已被 `.gitignore`（上游运行会自己生成），不要 track。

## `unlearn/` —— 本地 wrapper 脚本

OpenUnlearning 的启动命令偏长（Hydra override 一堆），`unlearn/*.sh` 是预配好的 shell wrapper：

| 脚本 | 用途 | 配置要点 |
|---|---|---|
| `wikitext_unlearn_sample.sh` | 单 triplet smoke-test（验证端到端能跑） | 深度配置（`max_steps=150`, `epoch=10`） |
| `wikitext_unlearn.sh` | 单 triplet 浅配置 | `max_steps=2`, `epoch=1`（历史 batch 用的浅配置，已知偏浅，见 STATE.md 阶段 ② 决策点） |
| `wikitext_unlearn_batch.sh` | 循环 100 个 triplet 的批量浅 unlearn | 同上浅配置 |
| `wikitext_unlearn_tofu_aligned.sh` | **当前主配置**，对齐 TOFU 默认超参 | `max_steps=3, epoch=5, bs=16, gas=4, lr=1e-5, optim=paged_adamw_32bit`（见 STATE.md 阶段 ② 训练配置表） |

`eval_wikitext_perplexity.py` 是**历史遗留**的 cross-PPL 脚本，真正的 PPL 评测属阶段 ③（`../2.extract-ppl/` 或等价目录）。本目录保留它只是为了方便单 ckpt 快速 sanity check。

### 重要的日志约定

- `logs_wikitext_unlearn_sample/` —— sample 脚本产生
- `logs_wikitext_unlearn_batch/` —— batch 脚本产生
- `logs_wikitext_unlearn_tofu/` —— TOFU-aligned 主配置产生
- `logs_*_bak/` —— 旧实验归档，不要删，但也不要当最新结果引用
- `batch_run.log` / `run.pid` —— 当前正在跑的 batch 的运行态

## 上下游契约

**读取阶段 ① 的**（每个 triplet）：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/train.json` → forget set（给 GradAscent forget loss）
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/validation.json` → retain set（给 retain-side loss）
- `test.json` **本阶段不读**（属阶段 ③ 的 cross-PPL 探针）

**写入给阶段 ③ 的**：
- `unlearn/saves/wikitext_unlearn_<tag>/wikitext_Llama-3.1-8B-Instruct_triplet_NNN_GradAscent/`
  - 4 分片 safetensors + `trainer_state.json` + tokenizer 文件
  - 单 ckpt ≈ 16 GB（bf16）；100 ckpt ≈ 1.6 TB（盘占见 STATE.md 警告）

## 对 Claude 的具体要求

1. **环境冲突以上游为准**：`open-unlearning/requirements.txt` 是权威，不要随意 pin 到更新版本的 `torch` / `transformers`。
2. **训练超参的权威位置**：Hydra config（`open-unlearning/configs/experiment/unlearn/wikitext/default.yaml` + CLI override）。改超参前先看 STATE.md 阶段 ② 的配置表是否需要同步更新。
3. **训练运行长耗时**：batch 跑 100 triplet ~1.5–2 h；不要在一个迭代里等到跑完，开 `run_in_background` + 日志监控。
4. **checkpoint 盘占极大**（16 GB × 100 = 1.6 TB），跑新配置前先确认盘够；必要时归档旧 `saves/wikitext_unlearn_*/`，不要随手 `rm -rf`。
5. **不要改上游源码绕过问题**：Hydra 找不到配置、Trainer 报错 —— 优先看 override、看 `open-unlearning/docs/hydra.md`、看上游 issue，最后才是本地 patch。
6. **max_steps 配置歧义**（见 STATE.md 阶段 ② 当前决策点）：写新脚本前先看 `wikitext_unlearn_tofu_aligned.sh` 为准，浅配置 `wikitext_unlearn*.sh` 是历史遗留。
