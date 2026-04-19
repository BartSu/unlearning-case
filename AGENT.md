# AGENT.md — Ralph Loop 任务规范

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
- 所有进度在 `PROGRESS.md` 中跟踪。每次迭代开始前必须阅读该文件。

## 循环约定
每次迭代必须：
1. 阅读 `PROGRESS.md` 以确定下一个未完成的任务。
2. 只选择一个任务，不要批量处理。
3. 实现 / 修复 / 验证该任务。
4. 运行相关脚本或测试确认其可用，并捕获输出。
5. 更新 `PROGRESS.md`：
   - 勾选已完成的条目。
   - 在 **迭代日志** 下追加一条记录，包含：迭代编号、任务、结果、产物、下一步。
6. **`z-doc/README-CN.md` / `slides.tex` / `slides-en.tex` 必须三份同步**（用户只看 slides 讲解）：
   - 任何内容 / 数字 / 叙述结构改动，三份须做镜像更新（slides-en 可保留英文措辞，但论证骨架与关键数字须一致）。
   - 新实验数字刷新时（`4.regression-predictor/audit/` 等 artefact 更新），三份中所有引用均须刷新。
   - 任一 `.tex` 改动后，在 `z-doc/` 下运行 `xelatex -interaction=nonstopmode slides.tex` 与 `xelatex -interaction=nonstopmode slides-en.tex`（各至少两遍以保证交叉引用），重新生成对应 PDF。
   - `slides.tex` / `slides-en.tex` / `README-CN.md` / `slides.pdf` / `slides-en.pdf` 一并纳入提交。
7. 使用简短的提交信息提交更改（禁止使用 `--no-verify`）。

## 任务选择规则
- 优先选择能够解锁下游阶段的任务。
- 如果同一任务连续失败两次，将其标记为 `BLOCKED` 并附上错误信息，然后转到下一个未被阻塞的任务。
- 除非被阻塞必须新增，否则不要自行发明 `PROGRESS.md` 中未列出的任务；若新增任务，需在迭代日志中说明理由。
- **当前瓶颈 = 阶段 2（unlearn 模型）**：100 个 triplet 中仅跑了 10 个代表，$n$ 扩到 100 是升级审计器 $\rho$ 置信度的必要前提。

## 完成标准
- `PROGRESS.md` 中的所有任务均已勾选完成。
- 完整流水线可在样例数据集上端到端运行。
- `make test`（或等价命令）通过。
- 输出：`<promise>COMPLETE</promise>`

## 安全约束
- 不得删除 `data/` 中的实验数据或模型检查点。
- 不得强制推送，不得改写 git 历史。
- 若 GPU/算力不可用，应将任务标记为 BLOCKED，而不是伪造结果。
- 绝不得捏造将被上报的结果。

## 退出机制
若连续 30 次迭代仍无进展（5 次迭代内无新完成项），写一份 `BLOCKERS.md` 总结阻塞情况，并输出 `<promise>COMPLETE</promise>` 终止循环。
