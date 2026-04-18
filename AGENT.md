# AGENT.md — Ralph Loop 任务规范

## 任务目标
迭代推进 LLM 遗忘（unlearning）研究流水线：数据准备 → 困惑度提取 → 几何分析 → 回归预测 corruption。

## 背景信息
- 仓库根目录：`/media/volume/llm/unlearning`
- 流水线各阶段以编号目录组织（`1.data-preparation/`、`2.extract-ppl/`、`4.regression-predictor/`）。
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
6. 若本次迭代产生了新的实验数字（即 `4.regression-predictor/audit/` 或其他 artefact 被刷新），必须同步：
   - 更新 `z-doc/README-CN.md` 与 `z-doc/slides.tex` 中所有引用的数字，使其与最新 `audit_summary.json` 一致。
   - 在 `z-doc/` 目录下运行 `xelatex -interaction=nonstopmode slides.tex`（至少两遍以保证交叉引用），重新生成 `z-doc/slides.pdf`。
   - 将 `slides.tex` 与 `slides.pdf` 一并纳入提交。
7. 使用简短的提交信息提交更改（禁止使用 `--no-verify`）。

## 任务选择规则
- 优先选择能够解锁下游阶段的任务。
- 如果同一任务连续失败两次，将其标记为 `BLOCKED` 并附上错误信息，然后转到下一个未被阻塞的任务。
- 除非被阻塞必须新增，否则不要自行发明 `PROGRESS.md` 中未列出的任务；若新增任务，需在迭代日志中说明理由。

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
