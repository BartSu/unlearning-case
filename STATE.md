# STATE.md — 当前研究状态快照

> 单一来源的「此刻状态」，供 Ralph 每次迭代开头快速对齐。
> 任务清单和完整历史在 [`PROGRESS.md`](PROGRESS.md)，迭代规则在 [`CLAUDE.md`](CLAUDE.md)。

**最近更新**：2026-04-23

---

## 一句话状态
五阶段流水线已打通到 $n=50$，审计器给出 L1/L2 显著、L3 边缘的排序信号；**正在等配置决策**以扩到 $n=100$。

## 关键数字（当前引用基准）
| 层 | geo-mean $r$ | 审计 $\rho$ | 95% CI | 审计 $R^2$（LOO） |
|---|---|---|---|---|
| L1 forget | **2.126×** | +0.624 | [−0.22, +0.93] | +0.443 |
| L2 locality | **1.491×** | +0.624 | [+0.05, +0.90] | +0.410 |
| L3 spillover | **1.283×** | +0.612 | [−0.01, +0.96] | +0.190 |

- 规模：$n=50$（5 triplet/cluster × 10 cluster；L1 n=2500 / L2 n=5000 / L3 n=122500）
- 产物：[`2.extract-ppl/corruption_summary.json`](2.extract-ppl/corruption_summary.json)、[`4.regression-predictor/audit/audit_summary.json`](4.regression-predictor/audit/audit_summary.json)

## 当前瓶颈（阻塞下游扩量）
**阶段 2（unlearn 模型）配置待决策**：
- `saves/wikitext_unlearn/triplet_001..100_GradAscent/` 100 个 checkpoint 已落盘（≈1.6 TB）。
- 但 batch 配置为 `max_steps=2 / num_train_epochs=1 / train_loss ≈ −2.4 / runtime ≈ 38 s`。
- 远**浅于**最初 sample 验证的 `max_steps=150 / epoch=10 / train_loss −735` 深度配置（相差 ~75 倍训练步数）。

**待用户决策**：
- **选项 A**：把 `max_steps=2` 视为有意的「轻触」正式配置 → 阶段 2 实质完成，补 cross-PPL 到 $n=100$ + 重跑审计器即可。
- **选项 B**：按 150-step 深度配置重跑 100 个 ckpt → 所有下游数字（审计器 $\rho$ / $R^2$ / MAE）同步重算。

## 正在进行 / 下一步
决策落定前可并行推进的免 GPU 任务：
- [ ] 重跑 `export_ppl_table.py` + `sanity_check_ppl.py` 刷新到 $n=50$ 的 `ppl_long.{parquet,jsonl}`（当前仍对应旧 $n=10$ 快照）
- [ ] 阶段 5 coverage 改进：signed projection / mutual-reachability 取代球覆盖（当前 $\rho=-0.42$ 方向反了）

决策落定后：
- [ ] 补剩余 50 triplet 的 cross-PPL → $n=100$
- [ ] $n=100$ 下重跑审计器，验证 $\rho$ 稳定性、收紧 L3 CI
- [ ] 加第二个 unlearner（NPO / GradDiff）验证跨算法稳健性

## 近期交付物
- **文档**：`z-doc/{slides.pdf (29 页), slides-en.pdf, README-CN.md}` 三份已同步到 $n=50$ 数字
- **Hero 图**：[`z-doc/figures/fig1_hero.pdf`](z-doc/figures/fig1_hero.pdf) + Figure 2 storyboard
- **复现包**：[`scripts/e2e_fresh_clone.sh`](scripts/e2e_fresh_clone.sh)（fresh clone 可从 §9 起端到端跑通）

## 风险 / 未决
- $n=50$ 下 L3 CI 下界仍触 0 附近 → L3 层的显著性依赖扩到 $n=100$
- 审计器目前只在 GradAscent 上验证，跨 unlearner 稳健性未测
- 单 base（Llama-3.1-8B-Instruct）→ 不能断言「corruption 是几何现象」是通用规律
