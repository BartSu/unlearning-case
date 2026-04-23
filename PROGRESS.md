# PROGRESS.md — Ralph Loop 进度跟踪

Ralph agent：每次迭代开始时阅读此文件，结束时更新此文件。

## 任务目标
迭代推进 LLM 遗忘（unlearning）研究流水线：
**数据准备 → unlearn 模型 → 困惑度提取 → 几何分析 → forget-set audit**。
最终目标：把 unlearning benchmarking 重构为 *forget-set 审计*（从"每次都跑 unlearn"到"先粗筛预警、再真跑 top-k"）。

## 背景信息
- 仓库根目录：`/media/volume/llm/unlearning`
- 五个流水线阶段与编号目录的对应关系：
  1. **数据准备**：`1.data-preparation/` —— WikiText-103 → HDBSCAN 10 簇 → 100 triplets（每簇 10 个）
  2. **Unlearn 模型**：`1.data-preparation/unlearn/` —— GradAscent，每 forget set 一个 checkpoint
  3. **困惑度提取**：`2.extract-ppl/` —— 10×10 cross-PPL，输出 L1/L2/L3 三层 ground truth
  4. **几何分析**：`3.feature-engineering/` + `4.regression-predictor/geometry/` —— forget-set 内禀几何特征（12 维）
  5. **Forget-set audit**：`4.regression-predictor/audit/` —— Ridge LOO，输出三层排序预测
- **研究定位**（与 `z-doc/slides.tex` 保持一致）：三层 corruption view 是 Act I 的观察结果；forget-set audit（"便宜的粗筛预警器"）是 Act II 的方法论提议。所有代码、文档、slides 的叙事都应围绕这两幕展开。

## 当前瓶颈
> 迁移到 [`STATE.md`](STATE.md)（关键数字 + 决策点 + 下一步，单一来源）。本文件专注「任务清单 + 迭代日志」。

## 文档同步规则
`z-doc/README-CN.md` / `slides.tex` / `slides-en.tex` **三份必须始终同步**（用户只看 slides 讲解）：
- 任何内容 / 数字 / 叙述结构改动，三份须做镜像更新（slides-en 可保留英文措辞，但论证骨架与关键数字须一致）。
- 新实验数字刷新时（`4.regression-predictor/audit/` 等 artefact 更新），三份中所有引用均须刷新。
- 任一 `.tex` 改动后，在 `z-doc/` 下运行 `xelatex -interaction=nonstopmode slides.tex` 与 `xelatex -interaction=nonstopmode slides-en.tex`（各至少两遍以保证交叉引用），重新生成对应 PDF。
- `slides.tex` / `slides-en.tex` / `README-CN.md` / `slides.pdf` / `slides-en.pdf` 一并纳入提交。

## 任务列表

### 阶段 1 — 数据准备（`1.data-preparation/`）
- [x] WikiText-103 → HDBSCAN 10 簇 → 100 triplets 已生成（`data/wikitext_hdbscan_triplets/triplet_001..100`）
- [x] 验证 `1.data-preparation/unlearn/wikitext_unlearn_sample.sh` 能在小样本上端到端运行（`saves/wikitext_unlearn_sample/.../triplet_001_GradAscent/` 150/150 step、epoch 10/10、4 分片 safetensors + `trainer_state.json` 齐全）
- [x] 恢复或替换已删除的 `1.data-preparation/unlearn/eval_wikitext_perplexity.py`（从 `c799c51~1` 恢复 698 行；路径更新为 `1.data-preparation/`；`py_compile` + `--help` 通过）
- [x] 在 `1.data-preparation/README.md` 中记录数据集 schema（triplet {train=forget, validation=retain, test=probe} × 50 条 `{"text": ...}`；100 triplet × 10 cluster 布局）

### 阶段 2 — Unlearn 模型（`1.data-preparation/unlearn/`）⚠️ **配置待决策**
- [x] 跑 10 个代表 triplet（`triplet_0{01,11,21,…,91}`）的 GradAscent checkpoint（深度配置：max_steps=150 / epoch=10，见 `saves/wikitext_unlearn_sample/`）
- [x] batch 跑满 100 个 triplet 的 GradAscent checkpoint（**浅配置**：max_steps=2 / epoch=1 / train_loss ≈ −2.4 / ~38 s 每个；落在 `saves/wikitext_unlearn/triplet_001..100_GradAscent/`）
- [ ] **决策点**：确认 max_steps=2 是有意"轻触"配置，或按 150-step 深度配置重跑 100 个 ckpt（若选重跑，所有下游数字需同步重算）
- [ ] 加第二个 unlearner（NPO / GradDiff）验证审计器跨算法稳健性

### 阶段 3 — 困惑度提取（`2.extract-ppl/`）
- [x] 对 10 个 unlearn ckpt 跑 10×10 cross-PPL → `wikitext_cross_metrics_detail.json`
- [x] `analyze_corruption.py` 产出 L1/L2/L3 三层 ground truth → `corruption_summary.json`（已扩到 n=50：5 triplet/cluster × 10 cluster，L1 n=2500 / L2 n=5000 / L3 n=122500；geo 2.126 / 1.491 / 1.283）
- [x] 以统一格式（parquet + jsonl）保存逐样本 PPL 长表（`export_ppl_table.py` → `ppl_long.{parquet,jsonl}`，**15000 行对应旧 n=10 快照**，layer ∈ {L1,L2,L3,L3_other}）
- [x] base vs unlearned PPL 分布 sanity-check 脚本（`sanity_check_ppl.py`，四项 invariant 在 n=10 快照上全 PASS：L1 geo=1.762x，L1>L2>L3 单调，base PPL 跨层一致，L1 100% 样本 ppl 上升）
- [ ] 重跑 `export_ppl_table.py` + `sanity_check_ppl.py` 刷新到 n=50 的 ppl_long（与最新 corruption_summary.json 对齐）
- [ ] 补剩余 50 个 triplet 的 cross-PPL 到完整 n=100（依赖阶段 2 配置决策）

### 阶段 4 — 几何分析（`3.feature-engineering/` + `4.regression-predictor/geometry/`）
- [x] 12 维 forget-set 内禀几何特征（variance / similarity / centroid / effective rank / isotropy）
- [ ] 为每个模型检查点提取隐状态几何特征（hidden-state geometry，目前只有 forget-set 嵌入几何）
- [x] 确认特征维度与回归输入匹配（16 几何特征 × 5000 行；LOGO CV by eval_triplet n=10 groups）
- [x] 验证 `4.regression-predictor/3.corruption_from_geometry.py` 能无错训练（Ridge R²=+0.17 / GB R²=+0.37 / RF R²=+0.28；L3-only RF R²=+0.25；产物 `geometry/{geometry_results.json,geometry_predictions.csv}` 已刷新）

### 阶段 5 — Forget-set audit（`4.regression-predictor/audit/`）
- [x] 运行 `4.audit_experiments.py` 并记录指标（LOO $n=10$：$\rho \approx 0.62$、L1 top-1 = storm）
- [x] Bootstrap 95% CI 的 $\rho$（$n=10$ 样本下）加到 audit_summary 与 slides（L1 [−0.22,+0.93]、L2 [+0.05,+0.90]、L3 [−0.01,+0.96]，n_boot=10000，seed=0）
- [x] 在留出 checkpoint 上报告 R² / MAE（n=10 LOO；Ridge 12 维几何。audit: L1 R²=+0.443 MAE=0.190；L2 R²=+0.410 MAE=0.121；L3 R²=+0.190 MAE=0.048；LOO-mean baseline R²=−0.235 全负，audit 在三层上均胜出）
- [ ] $n=100$ 下重跑审计器，验证 $\rho$ 稳定性（依赖阶段 2+3）
- [ ] 跨 unlearner 审计器迁移测试（依赖阶段 2）
- [ ] 改进 coverage：带符号投影 / mutual-reachability 取代球覆盖

### 横向 — 结果汇报（`z-doc/`）
- [x] 重新生成 `z-doc/slides.tex` 中引用的所有图表
- [x] 使用最新结果更新 `z-doc/README-CN.md`（已核对，数字与 audit_summary.json 一致）
- [x] 在全新克隆上完成最终端到端演练
- [x] 封面标题升级为 "From Three-Layer Corruption to Forget-Set Audit"
- [x] 新增英文 slides (`slides-en.tex` / `slides-en.pdf`)
- [x] 新增「审计器 = 预警器，不是替代品」定位帧 + README §5.4.1
- [ ] 新实验 ($n=100$ / 跨 unlearner) 出来后同步刷新三份 doc

## 迭代日志

### 迭代 15 — 2026-04-21（文件系统 vs PROGRESS.md 审计）
- 任务：核对 PROGRESS.md 所述"阶段 2 仅跑 10 个代表"与实际 `saves/` 目录的一致性（回应用户 "what's the result for stage 2?"）。
- 结果：partial —— 发现 PROGRESS.md 落后于实际状态，且揭出一个训练配置歧义。
  - **已落盘**：`1.data-preparation/unlearn/saves/wikitext_unlearn/` 下 `triplet_001..100_GradAscent/` 共 100 个完整 checkpoint（每个 4 分片 safetensors ≈ 16 GB，总计 ≈ 1.6 TB），时间戳 2026-04-19 17:20 前后。
  - **配置反差**：batch 下的 `trainer_state.json` 显示 `max_steps=2, num_train_epochs=1, train_loss ≈ −2.35, runtime ≈ 38s`；而最初验证用的 `saves/wikitext_unlearn_sample/triplet_001_GradAscent/` 为 `max_steps=150, epoch=10, train_loss=−735.46`。两者相差 ~75 倍训练步数 —— 不是规模扩展，而是明显不同的配置。
  - **下游已半同步**：`2.extract-ppl/corruption_summary.json`（Apr 19 21:03）已在其中 50 个 ckpt 上重算 `per_cluster`（001–095，每 cluster 5 个），n=2500/5000/122500；新 geo ratio L1=2.126 / L2=1.491 / L3=1.283 显著高于旧 n=10 的 1.762 / 1.290 / 1.158。
  - **下游未同步**：`ppl_long.parquet`（Apr 19 13:35，15000 行）仍对应旧 n=10 快照；`sanity_check_ppl.py` invariant（含 "L1 geo=1.762x"）未刷新；审计器 (`4.regression-predictor/audit/`) 所有数字仍基于 n=10。
- 产物：本 PROGRESS.md（阶段 2 checkbox 拆分 + 阶段 3 checkbox 增补 + "当前瓶颈" 段落改写）。未改代码 / 数据。
- 下一步（待用户决策）：
  - **配置决策**：max_steps=2 是否为有意"轻触"配置？若是 → 把当前 100 ckpt 认作 stage 2 完成，推进下游；若否 → 按 150-step 重跑 100 ckpt，所有下游重算。
  - 无论 A/B：将 `ppl_long.parquet` + `sanity_check_ppl.py` 的 invariant 刷新到与当前 `corruption_summary.json` 一致的 n。
  - z-doc 三份文档（`slides.tex` / `slides-en.tex` / `README-CN.md`）引用的 geo ratio (1.762/1.290/1.158) 与审计 ρ/R²/MAE 尚未反映 n=50 的新数字，延后到配置决策落定后一次性同步。

### 迭代 14 — 2026-04-19（reporter：同步 held-out R²/MAE 到 z-doc 三份）
- 任务：stage5-audit 迭代 11（held-out R²/MAE vs LOO-mean baseline）的文档同步。
- 结果：pass。`z-doc/README-CN.md` §5.3 在 bootstrap CI 表后新增 "Held-out R² / MAE" 对照表（audit vs baseline，L1 +0.44/−0.24，L2 +0.41/−0.24，L3 +0.19/−0.24；MAE 全线 audit 更低）+ 解读段落。`slides.tex` / `slides-en.tex` 「审计结果 / Audit results」表格加入 MAE 行（audit/baseline 并列），R² 行标注 baseline=−0.24，headline 改为 "三层 R² 全正、MAE 全线胜 baseline；n=10 CI 宽 ⇒ 排序级粗筛"。两份 PDF 各 xelatex 跑两遍（均 29 页）。
- 产物：`z-doc/{README-CN.md, slides.tex, slides-en.tex, slides.pdf, slides-en.pdf}`

### 迭代 13 — 2026-04-19（reporter：同步 bootstrap CI 到 z-doc 三份）
- 任务：stage5-audit 汇报迭代 8（bootstrap 95% CI）之后的文档同步。
- 结果：pass。`z-doc/README-CN.md` §5.3 新增 CI 小表（L1 [−0.22,+0.93]、L2 [+0.05,+0.90]、L3 [−0.01,+0.96]）+ 解读段落；`slides.tex` / `slides-en.tex` 在「审计结果 / Audit results」表格中插入 95% CI 行，headline 从 "三层 ρ≈0.62" 改为 "ρ≈0.62，但 n=10 下 CI 宽 ⇒ 排序级粗筛信号"。两份 PDF 各 xelatex 跑两遍（29 页）。
- 备注：`audit_summary.json` 同一次 commit 还追加了 `heldout_r2_mae` 字段（迭代 11 已记），slides 尚未纳入该字段，待后续单独汇报时再加入。
- 产物：`z-doc/{README-CN.md, slides.tex, slides-en.tex, slides.pdf, slides-en.pdf}`

### 迭代 12 — 2026-04-19（阶段 4：回归输入验证）
- 任务：阶段 4 子项 2、3 —— 确认特征维度匹配 + 验证 `4.regression-predictor/3.corruption_from_geometry.py` 可无错训练（免 GPU，读缓存 CSV）。
- 结果：pass。`geometry/corruption_geometry_features.csv` 提供 16 维纯几何特征 × 5000 行 (10 model_triplet × 10 eval_triplet × 50 test sample)，LOGO CV by eval_triplet 10 组。得分：mean baseline R²=−0.01；Ridge R²=+0.17；GB(300/3/0.05) R²=+0.37；RF(500/∞/3/0.5) R²=+0.28；cross-cluster（L3-only）RF R²=+0.25。RF 重要性头部：`forget_mean_pairwise_similarity` / `cos_sim_mean` / `forget_centroid_norm` / `forget_emb_variance_mean`。
- 产物：`4.regression-predictor/geometry/{geometry_results.json, geometry_predictions.csv}` 已刷新。
- 剩余：阶段 4 子项 1（hidden-state ckpt 几何）依赖阶段 2 扩到 100 ckpt，延后批量跑。
- 下一步：阶段 2 补完 90 triplet unlearn 后，批量在 10 → 100 ckpt 上提 hidden-state 几何特征。

### 迭代 11 — 2026-04-19（阶段 5 held-out R²/MAE + LOO-mean baseline）
- 任务：阶段 5 子项 "留出 checkpoint 报告 R² / MAE"
- 结果：pass。新增 `4.regression-predictor/6.heldout_r2_mae.py`，将 LOO 预测（Ridge alpha=1.0，12 维 forget-set 几何，训练 9 / 留 1）明确标注为 held-out 协议，补 MAE、并加 LOO-mean baseline 对照。数字（audit vs baseline）：
  - L1_forget:   R² +0.443 vs −0.235；MAE 0.190 vs 0.265；RMSE 0.225 vs 0.294
  - L2_locality: R² +0.410 vs −0.235；MAE 0.121 vs 0.154；RMSE 0.145 vs 0.171
  - L3_spillover: R² +0.190 vs −0.235；MAE 0.048 vs 0.061；RMSE 0.056 vs 0.067
  三层 audit 均胜 baseline；L3 绝对误差最小但主要因为 L3 尺度本身小（geo≈1.16）。写入 `audit_summary.json.heldout_r2_mae`。
- 产物：`4.regression-predictor/6.heldout_r2_mae.py`、`4.regression-predictor/audit/audit_summary.json`
- 下一步：交 reporter 同步 z-doc 三份；阶段 5 剩余 coverage → signed projection / mutual-reachability；n=100 / 跨 unlearner 依赖阶段 2。

### 迭代 10 — 2026-04-19（阶段 1 全部勾完）
- 任务：阶段 1 三个未勾选子项（sample 脚本端到端验证、恢复 `eval_wikitext_perplexity.py`、补 README schema）。
- 结果：pass。
  - **sample 脚本**：既有 `1.data-preparation/unlearn/saves/wikitext_unlearn_sample/wikitext_Llama-3.1-8B-Instruct_triplet_001_GradAscent/` 已是完整产物——`trainer_state.json` 显示 epoch=10/10、global_step=150/150、max_steps=150；4 分片 safetensors + tokenizer 齐全；train_loss −735.46（GradAscent，loss 越负越好），log 从 step 1 的 −2.25 单调下滑到 step 150 的 −1344.85，端到端跑通。
  - **eval 脚本**：从 commit `c799c51~1` 恢复 698 行原文件；repo 重命名后 `data-preparation → 1.data-preparation`，同步修正 3 处 `WIKITEXT_DIR_CANDIDATES` + 两处 docstring 示例；`py_compile` + `python eval_wikitext_perplexity.py --help` 均通过。
  - **README**：新增 `1.data-preparation/README.md`，记录 pipeline 摘要、文件布局、triplet schema（`{"text": ...}` 字段，train=forget/validation=retain/test=probe 三 split × 50 条）、`run_manifest.json` 关键参数（forget_size=50、triplets_per_domain=10、seed=42）、100 triplet × 10 cluster 的索引方式、与 stage 2/3 的下游契约。
- 产物：`1.data-preparation/unlearn/eval_wikitext_perplexity.py`（restored + path-patched）、`1.data-preparation/README.md`
- 下一步：阶段 1 全部完成；当前瓶颈仍在阶段 2（补 90 triplet unlearn）。

### 迭代 9 — 2026-04-19（阶段 3 产物标准化 + sanity-check）
- 任务：阶段 3 子任务 1+2。新增 `2.extract-ppl/export_ppl_table.py` 将嵌套 `wikitext_cross_metrics_detail.json` 展平为长表（parquet + jsonl），每行一条 (model_triplet, eval_triplet, split, sample_idx) 记录，含 base/unlearn loss+ppl+n_tokens、log_ppl_ratio、corruption layer（L1/L2/L3/L3_other）。新增 `sanity_check_ppl.py` 跑四项 invariant。
- 结果：pass。15000 行；L1=500, L2=1000, L3=4500, L3_other=9000。sanity check 全 PASS，geo-mean 与 `corruption_summary.json` 完全一致（1.762 / 1.290 / 1.158）。
- 产物：`2.extract-ppl/{export_ppl_table.py, sanity_check_ppl.py, ppl_long.parquet, ppl_long.jsonl}`
- 下一步：阶段 3 子任务 3（90 triplet cross-PPL）等 stage2-unlearn 出 ckpt 后启动。

### 迭代 8 — 2026-04-19（阶段 5 bootstrap 95% CI）
- 任务：阶段 5 子项 "Bootstrap 95% CI 的 ρ（n=10 下）"
- 结果：pass。新增 `4.regression-predictor/5.bootstrap_rho_ci.py`，对 `part2_audit_predictions.csv` 的 LOO (true, predicted) 对做 percentile bootstrap（n_boot=10000、seed=0），将 `bootstrap_rho_ci` 字段并入 `audit_summary.json`。结果：L1 ρ=+0.624 95% CI [−0.217,+0.926]；L2 ρ=+0.624 CI [+0.052,+0.899]；L3 ρ=+0.612 CI [−0.013,+0.963]。n=10 下 CI 宽、L1/L3 下界触到 0 附近，与 slides "预警器（排序粗筛）而非替代品" 叙事一致，且凸显扩 n=100 的必要性。
- 产物：`4.regression-predictor/5.bootstrap_rho_ci.py`、`4.regression-predictor/audit/audit_summary.json`
- 下一步：交棒 reporter 同步 z-doc 三份 slides / README-CN 的审计数字段落；阶段 5 剩余 "留出 ckpt R²/MAE" 与 "coverage 改 signed projection"；n=100 依赖阶段 2。


<!-- 在此追加条目，最新在最上方。格式：
### 迭代 N — YYYY-MM-DD
- 任务：<对应条目>
- 结果：<pass/fail/partial>
- 产物：<路径、commit 哈希>
- 下一步：<下一个任务 id>
-->

### 迭代 7 — 2026-04-19（论文定位升级 + slides-en + 流水线重定义）
- 任务：封面标题与叙事定位调整；新增英文版 slides；把 AGENT.md 流水线从四阶段扩展为五阶段（加入独立的 "unlearn 模型" 阶段）。非 PROGRESS 原列表项，作为方法论升级记录。
- 结果：pass。主要改动：
  - **slides**：封面 `\title` 改为 "From Three-Layer Corruption to Forget-Set Audit in LLM Unlearning"，subtitle 改为 "不跑 unlearn，只看 forget set 几何 ⇒ 排序级副作用预警"；论文定位帧同步。
  - **新增预警器帧**：`审计器 = 便宜的粗筛预警器`，明确"排序 / 粗筛 / 省算力"可做，"绝对数值 / 给 unlearn 打分 / 省掉最终 unlearn"不可做。
  - **新增 12 维特征解释帧**：分散度 / 相似度 / 位置规模 / 形状集中度四类。
  - **定义帧合并**：同心圆图 + L1/L2/L3 大白话短语 + `r = PPL_unlearned/PPL_base` 公式合并到一帧，放在 "一句话贡献" 之前。
  - **slides-en.tex**：完整英文版 29 页。
  - **README-CN.md**：标题与 slides 对齐；§5 改名 "先审计，再决定"；新增 §5.4.1 "预警器，不是替代品" 表；§11 论文定位 Title 同步。
  - **AGENT.md**：任务目标补 "unlearn 模型" 独立阶段；背景信息列出五阶段与目录映射；规则 6 扩展到三份 doc（加 slides-en）；任务选择规则加当前瓶颈 = 阶段 2。
- 产物：commit `25f37d0` (z-doc 三份同步)、`a9b7eba` (AGENT.md)；`z-doc/slides.tex`、`z-doc/slides-en.tex`、`z-doc/README-CN.md` 及两份 PDF。
- 下一步：阶段 2 扩 n（补 90 个 triplet 的 unlearn）；阶段 5 先把 bootstrap 95% CI 打出来（免 GPU，几分钟可做）；留出 checkpoint R²/MAE 报告。

### 迭代 6 — 2026-04-18（slides 可读性迭代，Ralph 10 轮）
- 任务：用户要求「slides 每页想表达的内容越简单越好，尤其三层定义」。非 PROGRESS 列表项，作为 Stage 5 叙述打磨记录。
- 结果：pass。10 轮共 9 次 commit（`dbebf4c..b7decb9`），改动集中在 `z-doc/slides.tex` + 重编 `slides.pdf`（28 页）：
  - **三层定义**（用户核心关切）拆成两帧：`同心圆图 + 箭头标注 + 距离轴` / `r=PPL_unlearned/PPL_base 具体例子 r=1.8`；另加一帧 `L1/L2/L3 在 10×10 矩阵里的位置` 作数据视角补充。
  - 一页一核心信息：一句话贡献、动机、C1+C2 headline、C3、审计问题/结果、几何-vs-文本表面、负结果、论文定位 均压成 headline+少量数字。
  - Limitations 压成「四个单」，Future Work 对齐解法；section 标题精简避免换行。
  - 数字未动，骨架未动，README-CN.md 无需同步（仅措辞简化，符合 AGENT.md 规则 6 的「精简化措辞」例外）。
- 产物：`z-doc/slides.tex`、`z-doc/slides.pdf`
- 下一步：阶段 4 留出 checkpoint R²/MAE；若要把审计从 n=10 扩到 n=100，补跑其余 90 个 triplet 的 unlearn + PPL + 几何。

### 迭代 5 — 2026-04-18
- 任务：阶段 5 结果汇报 —— 根据现有 audit 产物（n=10 代表 triplet，10 个 HDBSCAN cluster 各抽 1 个）重新生成 slides.tex 引用的三张图，并重新编译 slides.pdf。
- 结果：pass。`python z-doc/figures/make_figures.py` 读取 `4.regression-predictor/audit/{part1_corruption_profile,part1_per_sample_layers,part2_audit_predictions}.csv` 再生成 `fig_three_layer_decay.pdf` / `fig_per_forget_profile.pdf` / `fig_audit_scatter.pdf`，三张 PDF 均有差异并已更新。`xelatex slides.tex` 连跑两遍无错，`slides.pdf` 26 页 341 KB。关键数字与 audit_summary.json 完全一致：L1 geo=1.762×、L2 geo=1.290×、L3 geo=1.158×；LOO R² = 0.443 / 0.410 / 0.190；ρ ≈ 0.62；coverage ρ=−0.42。README-CN.md / slides.tex 叙述无需改动。
- 产物：`z-doc/figures/fig_*.pdf`、`z-doc/slides.pdf`
- 备注：当前 n=10 源于 batch unlearn 仅对 `triplet_0{01,11,21,…,91}` 10 个 cluster 代表跑了 checkpoint；其余 90 个 triplet 候选池已生成但未 unlearn，属阶段 1–3 后续工作范畴。
- 下一步：阶段 4 留出 checkpoint R²/MAE；若需将 n 从 10 扩到 100，需补跑其余 90 个 triplet 的 unlearn + PPL + 几何特征。

### 迭代 4 — 2026-04-18
- 任务：阶段 5 —— 在全新克隆上完成最终端到端演练
- 结果：pass。`git clone` 本仓至 `/tmp/fresh`，运行 `scripts/e2e_fresh_clone.sh` 顺序执行 analyze_corruption → corruption_from_geometry → audit_experiments → make_figures。为使得 fresh clone 能从 §9 步骤 1 起跑通（不重 unlearn），whitelist 并纳入了核心 artefact：`2.extract-ppl/{wikitext_cross_metrics_detail,corruption_summary}.json`、`4.regression-predictor/{audit,geometry}/{*.csv,*.json}`、`1.data-preparation/data/wikitext_hdbscan_triplets/{run_manifest.json, triplet_0[01-91]/}`（10 代表 triplet，合计 ≈1.2 MB）。fresh clone 输出的 L1/L2/L3 geo-mean、LOO R²/ρ、storm top-1、coverage ρ=−0.42 全部与主仓一致。
- 产物：`scripts/e2e_fresh_clone.sh`；`2.extract-ppl/.gitignore`、`4.regression-predictor/.gitignore`、`1.data-preparation/.gitignore` 更新；新增 artefact 与 triplet 数据纳入 git。
- 下一步：阶段 5 全部完成。后续推进阶段 4 的留出 checkpoint R²/MAE、阶段 1–3 的补完。

### 迭代 3 — 2026-04-18
- 任务：阶段 5 —— 重新生成 `z-doc/slides.tex` 中引用的所有图表
- 结果：pass。新增 `z-doc/figures/make_figures.py`，从 `4.regression-predictor/audit/` 产出三张 PDF 图（三层衰减 / per–forget-set profile / 审计 LOO 散点），并在 slides.tex + README-CN.md 中以镜像方式嵌入。顺带修复 `附录 — 复现主结果` 帧缺失 `[fragile]` 导致 verbatim 报错的历史问题。xelatex 双次编译通过，slides.pdf 26 页，`pdftotext` 已确认三个新帧标题出现。
- 产物：`z-doc/figures/{make_figures.py, fig_three_layer_decay.pdf, fig_per_forget_profile.pdf, fig_audit_scatter.pdf}`、`z-doc/slides.tex`、`z-doc/slides.pdf`、`z-doc/README-CN.md`
- 下一步：阶段 5 剩余条目（全新克隆端到端演练）；阶段 4 剩余条目（留出 checkpoint 的 R²/MAE 报告）

### 迭代 2 — 2026-04-18
- 任务：核对 `z-doc/README-CN.md` 与 `z-doc/slides.tex` 中的数字与当前 `audit_summary.json` 是否一致
- 结果：pass（两份文档中的 L1/L2/L3 geo-mean、>1.1×/>2× 占比、LOO R²/ρ、top-1=storm、coverage ρ=−0.42 全部与本次重跑结果一致，无需修改）
- 产物：无（文档已就位）
- 下一步：阶段 5 剩余条目（图表重生成、全新克隆端到端演练）

### 迭代 1 — 2026-04-18
- 任务：验证三层 knowledge corruption 视角 + 审计 forget dataset 无需 unlearn
- 结果：pass
- 产物：`4.regression-predictor/audit/{part1_corruption_profile.csv, part1_per_sample_layers.csv, part2_forget_features.csv, part2_audit_predictions.csv, part3_coverage.csv, audit_summary.json}`
- 关键数字：
  - Part 1（三层 ground truth，n=500/1000/4500）：L1 geo=1.762×（>1.1× 占 95.6%，>2× 占 33.2%）；L2 geo=1.290×（>1.1× 占 82.8%）；L3 geo=1.158×（>1.1× 占 61.6%，>2× 占 0.0%）。三层单调衰减成立，且 L3 未归零。
  - Part 2（LOO，n=10，仅 forget 集几何 12 维）：L1 R²=+0.443 ρ=+0.624 top1 ✓；L2 R²=+0.410 ρ=+0.624；L3 R²=+0.190 ρ=+0.612。
  - Part 3b：mean retain-coverage vs 真实 L3 spillover ρ=−0.423。
- 下一步：阶段 4 剩余条目（留出 checkpoint 的 R²/MAE 报告）；阶段 5 图表重生成

## 阻塞项

<!-- 将 BLOCKED 任务连同错误细节移到此处。 -->
