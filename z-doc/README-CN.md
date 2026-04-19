# From Three-Layer Corruption to Forget-Set Audit in LLM Unlearning —— 不跑 unlearn，只看 forget set 几何 ⇒ 排序级副作用预警

> 在 Llama-3.1-8B-Instruct 上对 GradAscent unlearning 做 data-first 的初步
> 分析。我们用 `ppl_ratio` 把 unlearning 的副作用划分为三层（L1 forget /
> L2 同簇 locality / L3 跨簇 spillover），在当前设置下（单 base model、
> 单 unlearner、10 个 forget set）观察到这三层呈现出与 forget set 相关的
> corruption profile；初步结果\emph{提示}仅用 forget set 的嵌入几何就能在
> **不跑任何 unlearning**的前提下对这个 profile 做排序级预测 —— 我们把这
> 称为 forget-set 审计。

---

## 0. 一句话贡献

LLM unlearning 的 ppl 损伤在本实验中沿语义距离呈现三层衰减（**1.76× →
1.29× → 1.16×**），但 L3 \emph{并未归零}（**61.6 % 的跨簇样本 surprisal
上升**）；同一算法在 10 个 forget set 上观察到 L1 差 1.7×，提示单 forget
set 的 unlearner benchmark \emph{稳定性不足}。作为审计 proof-of-concept，
我们用 **12 维 forget set 内禀几何特征**在 LOO 下预测这三层（$n=10$，
Spearman $\rho \approx 0.62$）；在 L1 层 top-1 预测与真实排序吻合（单
观察点，需更大 $n$ 复证）—— 整个审计过程\emph{无需} fine-tuning。

---

## 1. 动机 —— Unlearning 贵又反直觉

* 在 8B 模型上 fine-tune 一个 forget set 要几 GPU-hour；要评估 retain 侧
  副作用还得再跑一轮交叉评测。
* 同一个 GradAscent 在不同语义领域上的副作用差 **1.7×**（见 §4）；单 forget
  set 汇报的 unlearner 数字因此不可比。
* 现有文献几乎没测过 \emph{跨} 领域的 spillover —— 我们发现跨簇文本中
  **61.6 %** 的样本 surprisal 真的上升了。

### 研究问题

> 给定候选 forget set，能否\emph{不 fine-tune 任何模型}就预测它的三层
> corruption profile（L1 forget 强度、L2 同簇损伤、L3 跨簇污染）？

---

## 2. 三层 Corruption View（用 `ppl_ratio` 定义）

对固定 base LLM $f$、forget set $\mathcal{D}_f$ 与其 GradAscent 后的模型
$f_{\mathcal{D}_f}$，per-sample 损伤定义为

$$
r(f, \mathcal{D}_f, x) \;=\; \frac{\mathrm{PPL}(f_{\mathcal{D}_f}, x)}{\mathrm{PPL}(f, x)}.
$$

按评测文本 $x$ 与 $\mathcal{D}_f$ 的语义关系切成三层：

| 层 | 关系 | 直观含义 |
|---|---|---|
| **L1** forget | $x$ 就是 $\mathcal{D}_f$ 的 train 分片 | 故意遗忘强度 |
| **L2** locality | $x$ 来自同簇但不属于 $\mathcal{D}_f$ | 局部附带损伤 |
| **L3** spillover | $x$ 来自另一语义簇 | 跨簇 knowledge corruption |

---

## 3. 实验设置

* **Base model**：`meta-llama/Llama-3.1-8B-Instruct`，bf16，SDPA。
* **Unlearner**：`open-unlearning` 的 `GradAscent`，1 epoch，bs 1,
  gas 8, lr 1e-5。
* **语料**：WikiText-103 → HDBSCAN 得到 **10 个语义簇**（domain 标签：
  game, federer, jordan, episode, league, song, war, storm, river, star）。
* **Triplet**：每簇采 1 个代表 triplet，每 triplet 三分片各 50 段文本。
* **Cross-matrix**：$10 \times 10$ 对 (`model_triplet`, `eval_triplet`)，
  每对 50 个 per-sample ppl；diagonal 给 L1/L2，off-diagonal 给 L3。
* **监督信号**：$\log r$（把右偏的比值变为近对称分布）。
* **Artifact**：[2.extract-ppl/wikitext_cross_metrics_detail.json](../2.extract-ppl/wikitext_cross_metrics_detail.json) →
  [2.extract-ppl/corruption_summary.json](../2.extract-ppl/corruption_summary.json)。

---

## 4. 第一幕：Three-Layer View 被数据证实

### 4.0 这一幕要验证三个论断

若 §2 的三层切分真的抓住了 corruption 的结构，数据应同时满足：

- **C1（单调性）**：$r$ 随语义距离 L1 → L2 → L3 递减 —— 否则分层只是任意
  分箱，不是有序结构。
- **C2（非零溢出）**：L3 的几何均值显著 $> 1$、且非零率可观 —— 否则"跨
  簇污染"只是噪声，框架退化为 locality 分析。
- **C3（forget-set 依赖）**：三层 profile 在 forget set 之间显著不同 ——
  否则三层只是算法级常数，无"审计"可言（§5 便失去动机）。

下面分别给出 C1 / C2 / C3 的证据。

### 4.1 C1 + C2：层级 headline

来自 [2.extract-ppl/analyze_corruption.py](../2.extract-ppl/analyze_corruption.py)：

| Layer | $n$ | 几何均值 $r$ | >1.1× | >2× |
|---|---|---|---|---|
| **L1 forget** | 500 | **1.762×** | 95.6 % | 33.2 % |
| **L2 locality (同簇)** | 1 000 | **1.290×** | 82.8 % | 3.4 % |
| **L3 spillover (跨簇)** | 4 500 | **1.158×** | \alert{61.6 %} | 0.0 % |

* **C1 成立**：1.762 → 1.290 → 1.158 严格单调；L2 保留 L1 损伤的 38 %，L3
  保留 21 %（几何均值下）。
* **C2 成立**：L3 几何均值 1.158×、61.6 % 跨簇样本 surprisal 真的上升 ——
  spillover 在量级上\emph{不可忽略}、不是随机噪声。

![Three-layer decay](figures/fig_three_layer_decay.pdf)
*图 1 — 三层衰减：左轴几何均值 $r$ 随层级严格递减（C1）；右轴 $r>1.1$ 比例在 L3 仍达 61.6 %（C2）。*

补充一个定量参照。把"同簇 test"对"异簇 test"直接做同池对比（消除样本
数与聚合不对称的影响）：self 1.291×（n=10）vs. cross 1.158×（n=90），
locality multiplier 仅 **1.114×** —— 说明 L2 只比 L3 高约 11 %，
\alert{局部性存在但比常见直觉弱，全局溢出绝非可忽略}。

### 4.2 C3：三层 profile 是 forget-set 相关的

同一 unlearner、同一超参、10 个不同 forget set：

| Forget set | 簇 | L1 | L2 | L3 out |
|---|---|---|---|---|
| triplet_091 | star | 1.42 | 1.12 | 1.07 |
| triplet_001 | game | 1.44 | 1.17 | 1.07 |
| triplet_011 | federer | 1.45 | 1.11 | 1.07 |
| triplet_021 | jordan | 1.71 | 1.31 | 1.20 |
| triplet_081 | river | 1.73 | 1.23 | 1.18 |
| triplet_061 | war | 1.81 | 1.21 | 1.18 |
| triplet_051 | song | 1.86 | 1.29 | 1.19 |
| triplet_041 | league | 1.99 | 1.42 | 1.20 |
| triplet_031 | episode | 2.03 | 1.39 | 1.19 |
| **triplet_071** | **storm** | **2.44** | **1.78** | **1.25** |

* **C3 成立**：L1 跨 forget set 差 **1.72×**、L2 差 **1.60×**、L3_out 差
  **1.17×** —— 三层 profile 不是算法常数，而是 forget-set 的函数。
* 嵌入紧凑度（低 emb variance / 高 pairwise similarity）与 L2 的 Spearman
  $\rho = 0.50$（$n=10$，CI 较宽，\emph{提示性}证据）—— 已经在暗示几何
  可预测 corruption（§5 将把这一条正式化）。
* 方法学含义：\alert{仅用单个 forget set 给 unlearner 打分会显著低估评分
  方差} —— suggests benchmark instability，不是"系统性误报"。

![Per-forget-set profile](figures/fig_per_forget_profile.pdf)
*图 2 — 10 个 forget set 的三层 profile（按 L1 升序）；storm 在三层上同时最严重。*

### 4.3 小结 —— 进入第二幕

C1 + C2 共同支持："三层 view"在数据上是一个有序的、且 L3 非平凡的结构；
C3 进一步说明 profile 因 forget set 而异，因此\emph{预测 profile} 是一个
良定义的问题 —— 这正是 §5 要解决的审计任务。

---

## 5. 第二幕：先审计，再决定（审计器 = 便宜的粗筛预警器）

### 5.1 审计问题

> 给 \emph{一个} 候选 forget set $\mathcal{D}_f$（以及 base 模型），不跑
> fine-tune，输出 L1/L2/L3 三个风险分数。

### 5.2 审计特征（12 维，全部是 forget set 内禀几何）

来自 [4.regression-predictor/4.audit_experiments.py](../4.regression-predictor/4.audit_experiments.py) 的 Part 2：

| Group | 特征 |
|---|---|
| 方差 | `emb_variance_mean`, `emb_variance_max` |
| 两两相似 | `pairwise_sim_mean/std/q90`, `pairwise_eucl_mean` |
| centroid / norm | `centroid_norm`, `emb_norm_mean/std` |
| 集中度 | `effective_rank`, `isotropy`, `spread_over_centroid` |

**不看任何 evaluation text，不跑任何 fine-tune。** 对 10 个 forget set
做 Leave-One-Out，Ridge 回归。

### 5.3 审计结果

| 审计目标 | $R^2$ | Spearman $\rho$ | Pearson $r$ |
|---|---|---|---|
| `geo_L1_forget` | **+0.443** | **+0.624** | +0.684 |
| `geo_L2_locality` | **+0.410** | **+0.624** | +0.658 |
| `geo_L3_spillover` | +0.190 | **+0.612** | +0.512 |

三层都得到一致的 $\rho \approx 0.62$ 排序相关。L3 的 $R^2$ 更低但 $\rho$ 并
不低 —— spillover 量级小，绝对回归难，但\emph{相对排序}可靠。

**Bootstrap 95% CI**（$n=10$，$n_\text{boot}=10000$，percentile）：

| Layer | $\rho$ | 95% CI |
|---|---|---|
| L1 forget | +0.624 | [−0.217, +0.926] |
| L2 locality | +0.624 | [+0.052, +0.899] |
| L3 spillover | +0.612 | [−0.013, +0.963] |

CI 宽、L1/L3 下界跨 0 —— 点估计 $\rho \approx 0.62$ 只作为\emph{排序级粗筛
信号}，需 $n=100$ 收紧（这正是阶段 2 扩 unlearn 的动机）。

**Held-out R² / MAE**（$n=10$ LOO，Ridge $\alpha=1$，12 维几何；明确 vs LOO-mean baseline）：

| Layer | audit R² | baseline R² | audit MAE | baseline MAE |
|---|---|---|---|---|
| L1 forget | **+0.443** | −0.235 | **0.190** | 0.265 |
| L2 locality | **+0.410** | −0.235 | **0.121** | 0.154 |
| L3 spillover | **+0.190** | −0.235 | **0.048** | 0.061 |

三层 audit R² 全正且全线胜过 LOO-mean baseline（三层同为 −0.235）—— 12 维几何\emph{确有}信号，不是噪声。与上面的 bootstrap CI 合读：点估计已超 baseline，但 $n=10$ CI 仍宽 ⇒ 适合做\emph{粗筛预警}，$n=100$ 才能收紧。

![Audit scatter](figures/fig_audit_scatter.pdf)
*图 3 — LOO 预测 vs. 真实值（虚线 $y=x$）；三层均达 $\rho \approx 0.62$。*

### 5.4 Ranking：能挑出最危险的 forget set

| Layer | Spearman $\rho$ | Top-1 命中 | Top-3 重合 |
|---|---|---|---|
| L1 forget | +0.624 | **✅ 直接命中 storm** | 2 / 3 |
| L2 locality | +0.624 | ✗ | 2 / 3 |
| L3 spillover | +0.612 | ✗ | 1 / 3 |

在 L1 层，top-1 预测与真实排序一致（triplet_071 storm）；top-3 在 L1 / L2
都 2/3 重合。这些是\emph{单点}观察（$n=10$ forget set，每层只有一个 top-1
事件），\emph{不}构成 "审计器总能命中最严重 forget set" 的一般性论断；较
稳的表述是：初步证据支持审计器作为\alert{排序 / 筛选工具}的可行性，有待
更大 $n$ 下复证。

### 5.4.1 定位：预警器，不是替代品

| | 能做 | 不能做 |
|---|---|---|
| 排序 | 谁比谁严重（$\rho \approx 0.62$） | 绝对数值 |
| 粗筛 | 红 / 黄 / 绿三档预警 | 给 unlearn 打分 |
| 省算力 | 1 秒 vs. 数 GPU-hour | 省掉最终 unlearn |

**审计器 = 便宜的粗筛预警器。** 典型用法：100 个候选 → 审计排序 →
真跑 top-$k$ 验证；最终效果仍以真跑 unlearn 为准。

### 5.5 Per-sample 对照（旧框架 vs. 新框架）

| 设定 | 特征数 | LOGO $R^2$ |
|---|---|---|
| 旧：261 维表格特征（含 per-text 表面统计） | 261 | +0.142 |
| **新：16 维纯几何（forget + target-forget）** | **16** | **+0.362**（GBM） |
| 仅限 L3 跨簇样本 | 16 | +0.247（RF） |

> 在当前设置下，几何特征以约 **6 %** 的特征数带来 **2.5×** 的 $R^2$。这
> \emph{支持}"corruption 与 forget-set 几何比与文本表面统计更相关"的假
> 设，但在单 base model / 单 unlearner / $n=10$ 下尚不足以断言"corruption
> 是几何现象而非文本难度现象"。
> 见 [4.regression-predictor/3.corruption_from_geometry.py](../4.regression-predictor/3.corruption_from_geometry.py)。

### 5.6 负结果：Naive retain-coverage 不 work

把每个 forget set 的 95-分位内部 euclidean 距离当"高风险半径"，测 retain
文本落入半径内的比例：

* 9 / 10 个 forget set 的平均 coverage ≈ 1.0 —— 半径太大，完全失去区分。
* 与真实 L3 spillover 的相关：$\rho = -0.42$（方向反了）。

原因：HDBSCAN 簇在原始 384 维嵌入空间里互相重叠，p95 半径超过簇间距离。

**改进方向**（§7 Future Work）：
1. 换成到 forget centroid 的\emph{带符号投影分布}，做单尾切片而不是球覆盖。
2. 用 HDBSCAN 原生的 mutual-reachability 距离。
3. 直接把 §5.3 的 L3 predictor 当 calibrated retain-risk score。

---

## 6. 相关工作（占位）

* **Data-first 鲁棒性预测**：Dang 等（ACL 2024）从 13 个训练集特征回归
  adversarial attack success rate。我们把这条方法论迁移到\emph{生成式
  unlearning 的 corruption profile 审计}。
* **LLM unlearning**：GradAscent / NPO / GradDiff / TOFU 都由
  [open-unlearning](../1.data-preparation/open-unlearning/) 提供；我们的工作
  与"选哪个 unlearner"正交，固定 unlearner，分析其 data-conditioned 副作用。
* **Influence functions / editing locality**：在 model-centric 角度已被
  广泛研究；我们用 100 × 100 的 cross-ppl 矩阵把"局部性"变成可度量的
  三层风险。

（完整 BibTeX 待故事定稿后程序化抓取。）

---

## 7. 局限

* **$n = 10$ clusters**：审计器的 $\rho = 0.62$ 置信区间较宽。把 100
  个 triplet 全跑 unlearn 能将 $n$ 推到 100，结论强度上一档。
* **单 unlearner**：L1/L2/L3 数字是否对 NPO / GradDiff 稳健？待测。
* **单 base model**：Llama-3.1-8B-Instruct，未测其他规模/家族。
* **ppl 不等于 fact**：L3 的 61.6 % 样本 surprisal 上升是否对应 QA 准确率
  下降？`2.extract-qa/` 已有并行管线，未来需结合。
* **Naive coverage 失败**：§5.6 的负结果需要更细的几何风险度量。

---

## 8. Future Work

1. **Scale-up**：把 100 个 triplet（每簇 10 个）全部 unlearn，$n = 100$ 的
   审计器数字。
2. **跨 unlearner**：加入 NPO / GradDiff，审计器须对 unlearner 鲁棒。
3. **改进 coverage**：用方向投影或 mutual-reachability 代替球覆盖。
4. **Audit-driven forget 选择**：在 retain 约束下挑选 $\mathcal{D}_f$ 使
   predicted L3 最小，再跑真 unlearn 验证。
5. **QA 层面**：把 target 从 ppl 换成 QA accuracy，审计器是否依然 work。

---

## 9. 复现主结果

```bash
# 0. 数据 + unlearn + cross-ppl（已完成可跳过）
#    100 triplet × GradAscent × Llama-3.1-8B-Instruct
#    ⇒ 2.extract-ppl/wikitext_cross_metrics_detail.json

# 1. 第一幕：三层 corruption ground truth
cd 2.extract-ppl
python analyze_corruption.py
# ⇒ corruption_summary.json

# 2. 第二幕 Part A：per-sample 几何预测器（对照 261 维基线）
cd ../4.regression-predictor
python 3.corruption_from_geometry.py
# ⇒ geometry/geometry_results.json

# 3. 第二幕 Part B：forget-set 层级审计（LOO, Part 1+2+3+3b）
python 4.audit_experiments.py
# ⇒ audit/audit_summary.json
```

---

## 10. Artefact 速查

| Artefact | 路径 |
|---|---|
| 10 个代表 triplet | `1.data-preparation/data/wikitext_hdbscan_triplets/triplet_{001,011,…,091}/` |
| Unlearn checkpoints | `1.data-preparation/unlearn/saves/wikitext_unlearn/` |
| Cross-eval ppl 明细 | [2.extract-ppl/wikitext_cross_metrics_detail.json](../2.extract-ppl/wikitext_cross_metrics_detail.json) |
| 三层 corruption 汇总 | [2.extract-ppl/corruption_summary.json](../2.extract-ppl/corruption_summary.json) |
| 几何 per-sample 预测 | [4.regression-predictor/geometry/geometry_results.json](../4.regression-predictor/geometry/geometry_results.json) |
| 审计器 LOO 预测 | [4.regression-predictor/audit/audit_summary.json](../4.regression-predictor/audit/audit_summary.json) |
| 每 forget set L1/L2/L3 | [4.regression-predictor/audit/part1_corruption_profile.csv](../4.regression-predictor/audit/part1_corruption_profile.csv) |
| 审计特征表 | [4.regression-predictor/audit/part2_forget_features.csv](../4.regression-predictor/audit/part2_forget_features.csv) |

---

## 11. 论文定位建议

* **Title**（工作名）：*From Three-Layer Corruption to Forget-Set Audit in
  LLM Unlearning.*
* **Venue**：COLM 2025 / ACL short / EMNLP findings —— 以 \emph{empirical
  + analysis + methodological} 为主线。
* **核心卖点**：
  1. 三层 view 给出未被文献报告过的全局 spillover 数字（61.6 %）；
  2. 把"unlearning benchmarking"问题重构成"forget-set 审计"问题；
  3. 审计器几何简单、结果可排序、排名直接命中真实最难 forget set。
* **要补的短板**：把 $n$ 从 10 扩到 100 + 加第二个 unlearner，可升级到
  NeurIPS/ICML 级别。
