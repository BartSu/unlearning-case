# LLM Unlearning — 三层 Corruption + Forget-Set 审计

> 不跑 unlearn，只看 forget set 几何 ⇒ 排序级副作用预警。
> Llama-3.1-8B-Instruct · GradAscent · WikiText-103 · HDBSCAN × 10 簇 · $n=50$ forget set。

---

## 🗂️ 项目元文档（三份关键文件）

| 文件 | 作用 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | **Ralph Loop 迭代规则** —— agent 每次迭代的读写约定、任务选择规则、安全约束。 |
| [`STATE.md`](STATE.md) | **当前研究状态快照** —— 瓶颈、关键数字、下一步决策点。Ralph 每次迭代开头先读这个。 |
| [`PROGRESS.md`](PROGRESS.md) | **任务清单 + 迭代日志** —— 五阶段流水线的任务目标、目录映射、文档同步规则、历次迭代记录。 |

> 职责分工：`CLAUDE.md` 是**怎么干**，`STATE.md` 是**现在卡在哪**，`PROGRESS.md` 是**都干了啥 / 还要干啥**。

---

## 🎞️ 讲解材料（slides）

| 材料 | 用途 | 路径 |
|---|---|---|
| 🎞️ 中文 slides (PDF) | 演讲主用，29 页 | [`z-doc/slides.pdf`](z-doc/slides.pdf) |
| 🎞️ 英文 slides (PDF) | 给英文听众 | [`z-doc/slides-en.pdf`](z-doc/slides-en.pdf) |
| 📄 Deep-dive README (中文) | 讲稿底稿，所有数字出处 | [`z-doc/README-CN.md`](z-doc/README-CN.md) |
| 📎 slides 源文件 (tex) | 改 slides 用 | [`z-doc/slides.tex`](z-doc/slides.tex) / [`z-doc/slides-en.tex`](z-doc/slides-en.tex) |
| 🖼️ Hero 图（一图看懂） | 讲第一张图 | [`z-doc/figures/fig1_hero.pdf`](z-doc/figures/fig1_hero.pdf) |

更详细的讲解流程、关键数字出处、预想问答，见 [`z-doc/README-CN.md`](z-doc/README-CN.md)。
