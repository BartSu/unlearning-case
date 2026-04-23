# STATE.md — 当前研究状态快照

> 单一来源的「此刻状态」。
> 任务清单和完整历史在 [`PROGRESS.md`](PROGRESS.md)，迭代规则在 [`CLAUDE.md`](CLAUDE.md)。
---

## 数据准备（阶段 ①）

**状态**：✅ **冻结** —— 数据集、manifest、目录边界均已落定，下游（阶段 ②③）可直接消费。本阶段的规则 / 契约 / 硬性要求写在 [`1.data-preparation/CLAUDE.md`](1.data-preparation/CLAUDE.md)（子目录 CLAUDE）。

**目录边界**（2026-04-23 整理，commit `81e67ee`）：

```
1.data-preparation/
├── CLAUDE.md          # 子目录规则 + 下游契约（唯一文档入口）
├── .gitignore         # 仅白名单 10 个代表 triplet + manifest
└── data/wikitext_hdbscan_triplets/
    ├── run_manifest.json
    └── triplet_001 … triplet_100/{train,validation,test}.json
```

> 原 `1.data-preparation/open-unlearning/` 和 `1.data-preparation/unlearn/` 已搬到 [`2.train-unlearn/`](2.train-unlearn/)，本目录不再放训练 / 评估代码。

**来源与管线**（见 [`1.data-preparation/data/wikitext_hdbscan_triplets/run_manifest.json`](1.data-preparation/data/wikitext_hdbscan_triplets/run_manifest.json)）：

| 步骤 | 输出规模 | 关键参数 |
|---|---|---|
| WikiText-103 过滤 | 16 038 条文本 | 无特殊过滤 |
| 嵌入降维 | 384 维 | `reducer=null`（直接用原始嵌入的前 384 维） |
| HDBSCAN 聚类 | **10 簇 + 9 198 噪声** (57.4%) | `min_cluster_size=200, min_samples=5, metric=euclidean, cluster_selection_method=eom` |
| 簇规模分布 | 202 / 346 / 3 276（最小 / 中位 / 最大） | — |
| Triplet 生成 | **100 triplets = 10 domain × 10 per-domain** | `forget_size=validation_size=test_size=50, seed=42, triplets_per_domain=10, required_cluster_size=150` |

**Triplet schema**（每 triplet 三个 JSON 文件，每条 `{"text": ...}`）：

| 文件 | 研究语义 | 大小 | 用途 |
|---|---|---|---|
| `train.json` | **forget set** | 50 条 | unlearn 阶段喂给 GradAscent |
| `validation.json` | **retain set**（同簇邻居） | 50 条 | unlearn 阶段的 retain-side loss + cross-PPL 的 L2 probe |
| `test.json` | **probe**（同簇留出） | 50 条 | cross-PPL 的 L2/L3 evaluation |

> 三个 split 都来自**同一个 cluster**（同一个 domain），且**在同一 triplet 内互不相交**（不同 triplet 间允许文本重叠，详见 [`1.data-preparation/CLAUDE.md`](1.data-preparation/CLAUDE.md) §对下游的保证 / 不保证）。L3 spillover = 用**其他 triplet** 的 `test.json` 去打当前 unlearn 模型。

**每簇 10 个 triplets → 100 triplets**：cluster 标签 0..9，命名 `triplet_001..100`，前 10 个来自 cluster 0，11–20 来自 cluster 1 ... 依此类推。10 个 domain 名见 manifest，如 `game_yard_tech` (cluster 0)、`federer_white_open` (cluster 1)。

**git 跟踪范围**：仓库只 `git-track` 10 个代表 triplet（`triplet_0{01,11,21,31,41,51,61,71,81,91}`，即每簇第一个）+ `run_manifest.json`，用于 `n=10` audit reproduce；其余 90 个 triplet 存在于本地盘，阶段 ② batch 跑时按目录名直接索引。

**本阶段瓶颈**：无。如果下游需要改采样规则（例如 `triplets_per_domain` / `required_cluster_size`），**先开任务再重跑**，不要原地覆盖 `run_manifest.json`（见 `1.data-preparation/CLAUDE.md` 的 FROZEN 规则）。

---

## Unlearn 配置（阶段 ②）

**框架**：[`2.train-unlearn/open-unlearning/`](2.train-unlearn/open-unlearning)（HF Trainer 封装），entry = `src/train.py`，config = `unlearn.yaml` + `experiment=unlearn/wikitext/default`。
**脚本**：[`2.train-unlearn/unlearn/wikitext_unlearn_tofu_aligned.sh`](2.train-unlearn/unlearn/wikitext_unlearn_tofu_aligned.sh)

**基础设定**：
- Base：`Llama-3.1-8B-Instruct`（bf16）
- Method：`GradAscent`（首版唯一方法，NPO / GradDiff 作为后续跨算法验证）
- Attention：`sdpa`
- 硬件：**单卡 H100**

### 训练配置

| 维度 | 值 | 备注 |
|---|---|---|
| Forget / Retain 规模 | 50 / 50 | triplet schema 决定 |
| `num_train_epochs` | 5 | 数据被看 5 遍 |
| `per_device_train_batch_size` | 16 | |
| `gradient_accumulation_steps` | 4 | 累积 4 个 micro-batch 再更新一次权重 |
| `num_devices` | 1 | 单卡 H100 |
| `effective_batch` | 64 | = 16 × 4 × 1 |
| `learning_rate` | 1e-5 | |
| `lr_scheduler_type` | `linear` | HF Trainer 默认（未显式覆盖） |
| `warmup_steps` | 1 | = `max(1, steps_per_epoch)` |
| `optim` | `paged_adamw_32bit` | optimizer state 可 page 到 CPU |
| `weight_decay` | 0.01 | |
| `bf16` | True | 混合精度 |
| `max_steps` | 3 | 硬卡权重更新次数；open-unlearning 的 dataloader per-epoch step 行为未知，用 max_steps 钉死 |
| `samples_seen` | 250 | = 50 × 5 |

> **显存**：Llama-3.1-8B bf16 + `BS=16 × seq_len=500` 在单卡 H100 80GB 上可能逼近上限（模型 16 GB + paged_adamw optimizer 可 page 到 CPU + 激活 ~20 GB）。若 OOM 降到 `BS=8 GAS=8` 或 `BS=4 GAS=16`（effective_bs 均 = 64，优化等价）。

### 跑满 10 × 10 数据集（100 triplets）的时间预算

单 triplet 的主要开销：
- **Model load**（从磁盘加载 Llama-3.1-8B bf16 16 GB checkpoint）：H100 SSD 下 ~15–25 s
- **训练**（3 step × GAS=16 × BS=4 × (forget+retain) ≈ 384 sample forward+backward）：H100 bf16 大约 **5–10 s**
- **Tokenization / dataloader / hydra init**：~5–10 s
- **Save checkpoint**（~16 GB bf16）：~10–15 s

**单 triplet 总耗时估**：**45–70 s**
**100 triplet 全量**：**~1.5–2 h**（保守上界 3 h，含磁盘 IO 抖动）

写满 100 个 Llama-8B ckpt 需要 **~1.6 TB** 存储，与现有 `saves/wikitext_unlearn/` 同量级。如果保留现有浅 ckpt + 新 TOFU ckpt，盘占倍增，必要时先挑一个归档/删除。
