# PROGRESS.md — Ralph Loop 进度跟踪

Ralph agent：每次迭代开始时阅读此文件，结束时更新此文件。

## 任务列表

### 阶段 1 — 数据准备
- [ ] 验证 `1.data-preparation/unlearn/wikitext_unlearn_sample.sh` 能在小样本上端到端运行
- [ ] 恢复或替换已删除的 `1.data-preparation/unlearn/eval_wikitext_perplexity.py`
- [ ] 在 `1.data-preparation/README.md` 中记录数据集 schema

### 阶段 2 — 困惑度提取
- [ ] 在样例 unlearned 模型上运行 `2.extract-ppl/analyze_corruption.py`
- [ ] 以统一格式（parquet 或 jsonl）保存逐 token PPL 输出
- [ ] 增加对比基线与 unlearned PPL 分布的 sanity-check 脚本

### 阶段 3 — 几何特征
- [ ] 为每个模型检查点提取隐状态几何特征
- [ ] 确认特征维度与回归输入匹配

### 阶段 4 — 回归预测器
- [ ] 验证 `4.regression-predictor/3.corruption_from_geometry.py` 能无错训练
- [x] 运行 `4.regression-predictor/4.audit_experiments.py` 并记录指标
- [ ] 在留出检查点上报告 R² / MAE

### 阶段 5 — 结果汇报
- [x] 重新生成 `z-doc/slides.tex` 中引用的所有图表
- [x] 使用最新结果更新 `z-doc/README-CN.md`（已核对，数字与 audit_summary.json 一致）
- [x] 在全新克隆上完成最终端到端演练

## 迭代日志

<!-- 在此追加条目，最新在最上方。格式：
### 迭代 N — YYYY-MM-DD
- 任务：<对应条目>
- 结果：<pass/fail/partial>
- 产物：<路径、commit 哈希>
- 下一步：<下一个任务 id>
-->

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
