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
- [ ] 运行 `4.regression-predictor/4.audit_experiments.py` 并记录指标
- [ ] 在留出检查点上报告 R² / MAE

### 阶段 5 — 结果汇报
- [ ] 重新生成 `z-doc/slides.tex` 中引用的所有图表
- [ ] 使用最新结果更新 `z-doc/README-CN.md`
- [ ] 在全新克隆上完成最终端到端演练

## 迭代日志

<!-- 在此追加条目，最新在最上方。格式：
### 迭代 N — YYYY-MM-DD
- 任务：<对应条目>
- 结果：<pass/fail/partial>
- 产物：<路径、commit 哈希>
- 下一步：<下一个任务 id>
-->

## 阻塞项

<!-- 将 BLOCKED 任务连同错误细节移到此处。 -->
