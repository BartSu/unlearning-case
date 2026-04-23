# CLAUDE.md

## 文档职责
- [`STATE.md`](STATE.md) —— **当前状态快照**（瓶颈、关键数字、下一步决策点）。Ralph 每次迭代开头**先读这个**。
- [`PROGRESS.md`](PROGRESS.md) —— **任务清单 + 迭代日志**。任务目标、流水线阶段、目录映射、文档同步规则在顶部维护。
- [`README.md`](README.md) —— 项目首页导航（给人看，不给 Ralph 看）。

## 循环约定
每次迭代必须：
1. 阅读 `STATE.md` 获取当前瓶颈，再翻 `PROGRESS.md` 任务列表挑一个未完成、未被阻塞的任务。
2. 只选择一个任务，不要批量处理。
3. 实现 / 修复 / 验证该任务。
4. 运行相关脚本或测试确认其可用，并捕获输出。
5. 更新 `PROGRESS.md`：
   - 勾选已完成的条目。
   - 在 **迭代日志** 下追加一条记录，包含：迭代编号、任务、结果、产物、下一步。
6. 若本迭代改变了瓶颈、关键数字或决策点，同步更新 `STATE.md`（保持其与 `PROGRESS.md` 顶部一致）。
7. 遵循 `PROGRESS.md` 中的 **文档同步规则**（`z-doc/` 下 README-CN / slides / slides-en 三份须保持一致，任一 `.tex` 改动后重编 PDF）。
8. 使用简短的提交信息提交更改（禁止使用 `--no-verify`）。

## 子目录 CLAUDE.md 规则

**任何时候要在某个子目录下做事（读 / 改 / 跑脚本 / 补产物），必须先查该目录**及其到仓库根的父链上每一级**是否存在 `CLAUDE.md`；若存在，全部加载进 context 再动手**。加载顺序：**根 → 叶**（根的约定是基础，叶子的约定是最特定的）。冲突时，叶子（更接近目标文件）优先。

当前子目录 CLAUDE.md 清单：

| 路径 | 作用 |
|---|---|
| `./CLAUDE.md` | 项目级循环约定（本文件） |
| `1.data-preparation/CLAUDE.md` | 阶段 ① 数据冻结契约 / schema |
| `2.train-unlearn/CLAUDE.md` | 阶段 ② unlearn 配置 + **新机器 env bootstrap**（HF cache / CUDA_HOME / hydra `++`） |
| `3.inference/CLAUDE.md` | 阶段 ③ cross-PPL + 踩过的 glob bug / baseline 复用规则 |
| `4.feature-engineering/CLAUDE.md` | 阶段 ④ 特征与 ckpt 解耦 / 262 维 ablation store 定位 |
| `5.audit/CLAUDE.md` | 阶段 ⑤ `parents[2]` 路径规约 / 12 维 headline vs 262 维旁路的角色分工 |

**读子目录 CLAUDE.md 必须做的**：
- 其「对 Claude 的具体要求」和「已踩过的坑」——是前人调试后留的硬约束，**不要再撞一次**
- 其「上下游契约」——改动跨目录时必须同时兼容，不要单边打破
- 其「Smoke test」——动手前先跑一遍 sanity，不要盲改

**改动子目录时**：若你的改动改变了该目录的**脚本入口 / 路径常量 / 产物 schema / 环境要求**，必须**同步更新**该目录的 `CLAUDE.md`（「已踩过的坑」加新条或「对 Claude 的要求」加新约束）。CLAUDE.md 与代码要共演化，不要让文档滞后于代码导致下次误导。

**新增子目录 / 脚本时**：若目录下累计 > 2 个脚本或含独立产物 / 依赖独立环境，应为其新建 `CLAUDE.md`（模板：目录职责 / 子目录-脚本 / 上下游契约 / 对 Claude 的要求 / 已踩过的坑 / Smoke test 六段），参考 [`2.train-unlearn/CLAUDE.md`](2.train-unlearn/CLAUDE.md) 的结构。

## 任务选择规则
- 优先选择能够解锁下游阶段的任务（参照 `STATE.md` 的「当前瓶颈」标注）。
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