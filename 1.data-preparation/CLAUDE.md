# CLAUDE.md —— 阶段 ①「数据准备」

本目录的唯一产物是 **WikiText-103 HDBSCAN triplets 数据集**，供阶段 ② unlearn 训练和阶段 ③ cross-PPL 评测消费。

根目录约定（迭代循环、任务选择、文档同步、安全约束）见 [`../CLAUDE.md`](../CLAUDE.md)。本文件只补充**本阶段**专有的规则。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 持有已冻结的 triplet 数据集 + provenance manifest | 训练 / unlearn / 评估（属阶段 ②③） |
| 为下游提供稳定 schema 和命名 | 重跑聚类 / 重抽样（见下方「已冻结」） |
| 记录管线参数到 `run_manifest.json` | 修改 manifest 的历史字段 |

## 已冻结（FROZEN）

下列事实在 `data/wikitext_hdbscan_triplets/run_manifest.json` 中已存档，**不要重跑、不要改动**：

- WikiText-103 过滤：16 038 条
- 降维：`reducer=null`，384 维（原始嵌入前 384 维）
- HDBSCAN：`min_cluster_size=200, min_samples=5, metric=euclidean, cluster_selection_method=eom` → 10 簇 + 9 198 噪声 (57.4%)
- Triplet 采样：`forget_size=validation_size=test_size=50, triplets_per_domain=10, required_cluster_size=150, seed=42` → 100 triplets

如果下游发现需要改采样规模 / 规则，**先在 `PROGRESS.md` 开任务说明动机**，再讨论是否重跑。**不要**就地覆盖 `run_manifest.json`。

## 文件清单

```
1.data-preparation/
├── CLAUDE.md                           # 本文件（阶段 ① 唯一文档入口）
├── .gitignore                          # 只白名单 10 个审计代表 triplet + manifest
└── data/
    ├── scripts/                        # 数据生成管线代码（见下方章节）
    │   ├── 0.data_download.py  …  8.qa.py
    │   └── _hdbscan_pipeline_utils.py
    └── wikitext_hdbscan_triplets/
        ├── run_manifest.json           # provenance（上游路径 + 所有超参 + 每 triplet 元信息）
        └── triplet_001 … triplet_100/
            ├── train.json              # forget set      (50)
            ├── validation.json         # retain set      (50)
            └── test.json               # held-out probe  (50)
```

其他 `data/wikitext_*` 子目录（`wikitext_raw / filtered / embeddings / reduced / clusters_hdbscan*`）是 `data/scripts/` 跑出来的**上游生成物**，本仓库 `.gitignore` 掉、不进 git，存档路径写在 `run_manifest.json.source` 里，不保证能从仓库内部重生成。

## 数据生成管线代码（`data/scripts/`）

10 个脚本 + 1 个 utils，和 manifest.source 里的上游路径一一对应。**scripts 是工具，manifest 是产物**，「已冻结」规则约束的是产物，不是工具。

| 脚本 | 产物 | manifest.source 对应键 |
|---|---|---|
| `0.data_download.py` | WikiText-103 原始文本 | `wikitext_raw/`（未列入 source） |
| `1.filter.py` | 过滤后的文本 + offsets | `filtered_texts_jsonl`, `filtered_offsets_npy` |
| `2.embed.py` | sentence embeddings | `wikitext_embeddings/`（未列入 source） |
| `3.reduce_dimension.py` | 降维向量（此管线 `reducer=null`） | `reduced_vectors_path` |
| `4.cluster.py` | HDBSCAN 标签 + 距离 | `cluster_labels_path`, `cluster_distances_path` |
| `5.summarize.py` | 簇关键词 / domain 名 | `cluster_summary_json` |
| `6.export.py` | 簇-样本映射 csv | `cluster_assignments_csv`, `export_manifest_json` |
| `7.generate_triplet.py` | **本目录唯一持久产物** | `data/wikitext_hdbscan_triplets/` 全部 |
| `8.qa.py` | manifest QA / 数据健康检查 | — |
| `_hdbscan_pipeline_utils.py` | 公共工具 | — |

**重跑规则**（「已冻结」的实操补充）：
- 不要为了"试新配置"在本地直接跑脚本、把新 manifest 覆盖旧的 —— 会悄悄失去 provenance。
- 要重跑，先在 `PROGRESS.md` 开任务、备份旧 manifest、在**新目录**下跑，产出新 manifest，再在 STATE.md 里切换指向。
- manifest.source 里的路径是绝对路径（`/media/volume/llm/unlearning/data-preparation/data/...`），换机器重跑前先核对这些路径是否仍有效。

## `.gitignore` 白名单策略

仅把「每簇第一个 triplet」进 git：`triplet_0{01,11,21,31,41,51,61,71,81,91}` —— 这是 `n=10` audit reproduce 用的子集。其余 90 个 triplet 存在于本地磁盘但不追踪。全量 100 triplets 在阶段 ② batch 跑，不依赖 git。

## Triplet schema（下游契约）

每个 `triplet_NNN/{train,validation,test}.json` 是一个 JSON **list**，元素为 `{"text": "..."}`。三个 split 都来自**同一个 HDBSCAN 簇**（同一个 domain），互不相交。

| 文件 | 研究语义 | 阶段 ② / ③ 用途 |
|---|---|---|
| `train.json` | forget set | unlearn 的 GradAscent forget loss |
| `validation.json` | retain set（同簇邻居） | unlearn 的 retain-side loss；cross-PPL 的 L2 probe |
| `test.json` | held-out probe | cross-PPL 的 L2 / L3 evaluation |

**命名约定**：cluster 标签 0..9，`triplet_001..010` 属 cluster 0，`011..020` 属 cluster 1 … 依此类推。domain 字符串（如 `game_yard_tech`）由 manifest 中的簇关键词决定，下游**不要**依赖它做语义判断，只作为人可读标签。

## 对下游的保证 / 不保证

**保证**：
- `run_manifest.json.triplets[i]` 的字段集合稳定（`name, cluster_label, domain, domain_triplet_index, cluster_size, forget/validation/test_size, unused_cluster_samples_per_triplet`）。
- 三个 split 的大小恒为 50 / 50 / 50。
- 同一 triplet 的三个 split 来自同一 cluster、互不相交。

**不保证**：
- 不同 triplet 之间 `test.json` 允许来自同簇不同样本（L3 spillover 的语义来源）；**不**保证 triplet 之间文本完全不重复，但 seed 固定。
- domain 字符串不保证稳定（关键词算法可能变）—— 下游请优先用 `cluster_label` 而非 `domain`。

## 对 Claude 的具体要求

1. **不要**删除或覆盖 `data/wikitext_hdbscan_triplets/` 下任何文件，也不要改 `run_manifest.json`。
2. **不要**在本目录里写训练 / 评估代码 —— 走错阶段。训练放 `../2.train-unlearn/`，评估放 `../2.extract-ppl/`（或对应阶段目录）。
3. 如果 manifest 和 STATE.md 的数字对不上，**以 manifest 为准**，然后更新 STATE.md，而不是反过来。
4. 如果真的要重跑聚类 / triplet 采样，先开任务、先备份旧 manifest、再在新目录下生成，**不要原地覆盖**。
5. 需要验证数据健康度时，用 `run_manifest.json` 里的字段 + `json.load` 抽查 split 长度和 schema，不需要重算聚类。
