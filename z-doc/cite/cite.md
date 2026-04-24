# Related Work

本文件记录与本工作**直接相关**的文献 —— 启发本工作的研究视角，或作为本工作的工程基座。PDF 存同目录下同名文件。新增条目时请保持格式：**Full citation → Summary → How it inspires us → What we differ**，并把 PDF 下到本文件夹。

---

## 1. A Curious Case · forget-set audit 的视角来源

**Dang, Cuong; Dung D. Le; Thai Le.**
*A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models.*
arXiv:2402.11469, July 2024.
[`./a-curious-case.pdf`](./a-curious-case.pdf)

### Summary
他们指出 fine-tuned transformer model 的 adversarial robustness 并非纯 model-dependent，而是**训练数据本身的可度量属性**。传统做法 `model-first`：在每个 fine-tuned model 上跑 adversarial generation + evaluation，慢。他们做 `data-first`：从训练集本身提**数据侧特征**，用一个 lightweight regressor 就能预测该 model 被 adversarial perturb 后的 ASR —— 比 model-first 快 **30×–193×**。

### How it inspires us
**"不跑下游评测，只看数据侧特征就能预测下游行为"** 这个方法论范式是我们 forget-set audit 的直接原型。具体迁移：

| 他们（adversarial robustness） | 我们（unlearning corruption） |
|---|---|
| Fine-tuning dataset | Forget set $\mathcal{D}_f$ |
| 数据侧特征 → 预测 ASR | 12 维 forget-set 内禀几何 → 预测 L1 / L2 / L3 corruption |
| 30×–193× 加速 adversarial eval | 把 unlearn + N×N cross-PPL 的 GPU-hour 换成秒级几何推理 |

### What we differ
- 任务从 **adversarial robustness** 换到 **unlearning side-effect**，两者的 ground truth 机制完全不同（前者来自 attack loss，后者来自 PPL ratio 三层衰减）。
- 他们关心 "fine-tune 后模型对攻击有多脆弱"；我们关心 "unlearn 后**无关**数据被伤多少" —— 下游语义不同。
- 他们给**绝对**回归（ASR 数值）；我们定位为**排序粗筛**（ranking audit），坦诚 R²/ρ 在 n=10 下 L3 仍弱，paper 里明确"预警器，不是替代品"。

---

## 2. Probing Knowledge Holes · adjacent / latent corruption 的现象来源

**Ko, Myeongseob; Hoang Anh Just; Charles Fleming; Ming Jin; Ruoxi Jia.**
*Probing Knowledge Holes in Unlearned LLMs.*
arXiv:2511.00030, October 2025.
[`./knowledge_hole.pdf`](./knowledge_hole.pdf)

### Summary
他们观察到：近期 unlearning 方法在 **standard benchmark** 上看起来无害（utility metrics 基本不降），但模型内部实际留下了 **knowledge holes** —— 被 unlearn 目标**附近**的知识被悄悄抹掉了，只是 benchmark 照不出。论文提出 probing 方法主动暴露这些洞。

### How it inspires us
他们把 "unlearn 的真实副作用**不在目标本身**、而在其**相邻 / 潜在**知识" 这个现象推到方法论中心 —— 这正是我们三层 corruption view 里的 **L2 locality** 和 **L3 spillover** 要量化的东西。具体关系：

- **我们的 L2（locality）** ≈ 他们的 adjacent knowledge hole：同簇语义邻居被伤。
- **我们的 L3（spillover）** 比他们更远：跨话题簇的知识也被抬 PPL（~1.2×），他们主要 probe adjacent。
- **Framing 差异**：他们 probe unlearn **之后** 的 model；我们 audit unlearn **之前** 的 forget set 几何 —— 是 "Ko et al. 的问题被发现后，我们主张在事前预警"。

### What we differ
- 他们**实证观察现象**（knowledge holes 存在），本工作把它**量化成三层单调衰减** + **可从 forget-set 几何预测**。
- 他们的 probing 仍然需要 unlearn 完成；我们的 audit 不需要真跑 unlearn。
- 他们的 evaluation 是 QA correctness；我们用 PPL ratio（QA-based 评测在 `3.inference/extract-qa/` 作为未来并行视角）。

---

## 3. OpenUnlearning · 本工作使用的 code base

**Dorna, Vineeth; Anmol Mekala; Wenlong Zhao; Andrew McCallum; Zachary C. Lipton; J. Zico Kolter; Pratyush Maini.**
*OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics.*
arXiv:2506.12618, November 2025 (v2).
[`./open-unlearning.pdf`](./open-unlearning.pdf)

### Summary
统一 LLM unlearning benchmarking 框架：集成 **13 个 unlearning 方法**（GradAscent / NPO / GradDiff / RMU / DPO / ...）+ 16 个 evaluation metric + 3 个 benchmark（TOFU / MUSE / WMDP），公开 450+ checkpoint。核心是 Hydra 配置化的 `src/train.py` + `configs/` 结构，trainer 抽象让新算法只需加一个 subclass。

### How we use it
本工作**不是**对 OpenUnlearning 的 intellectual extension，而是**直接使用它**作为阶段 ② 的 unlearn 实现。vendored 到 [`../../2.train-unlearn/open-unlearning/`](../../2.train-unlearn/open-unlearning/) 下：

- 直接复用其 `GradAscent` trainer 实现（后续验证 NPO / GradDiff 也会从它的 `configs/trainer/` 拉）
- Hydra config override 调 TOFU-aligned 超参（`max_steps=5 / bs=16 gas=4 / lr=1e-5 / paged_adamw_32bit`，详见 `2.train-unlearn/CLAUDE.md`）
- **不改**它的核心源码，只通过 config override + wrapper shell 调用；任何 config 不能表达的需求（`max_steps` / `warmup_steps` / `lr_scheduler_type` 不在 struct 里）以 `++trainer.args.X=v` force-override 表达

### What we contribute on top
- OpenUnlearning 主要关心 **per-benchmark score**（TOFU utility / forget quality 等），benchmark 是**固定 forget split**。
- 我们在它之上做**数据侧研究**：换 forget set 看 corruption 怎么变、forget set 几何怎么预测 corruption。这是**选 forget set** 这层的工作，与其**正交互补**。
- 若本工作接入其 lm-eval-harness / WMDP 等 QA 评测，就能把 PPL ratio + QA accuracy 两个视角并报。

---

## 其它（not yet cited, but relevant）

- **TOFU benchmark** (Maini et al. 2024, arXiv:2401.06121) —— OpenUnlearning 的基础数据集 splits；paper 扩到 TOFU 而不仅 WikiText 时引用。
- **MUSE benchmark** (Shi et al. 2024, arXiv:2407.06460) —— Machine Unlearning Six-Way Evaluation，影响 "unlearn 是否真实" 的 metric 选择。
- **RMU** (Li et al. 2024, arXiv:2403.03218) —— representation-engineering unlearning，作为跨 unlearner 验证候选之一。
- **Influence functions / model editing locality** (多篇) —— 与本工作的 **data-centric 三层风险 view** 形成对照（他们是 model-centric），slides §Related Work 已提，paper 扩展时补。

---

## 维护守则

1. 新增文献 **必须**给出：full citation（authors, title, venue, year, arXiv id）+ PDF 放本目录。
2. 每条 3 段：Summary（3-5 句客观描述）/ How it inspires/uses us（本工作怎么受其启发或使用它）/ What we differ（本工作的具体 positioning）。
3. 不追求 completeness，只收**真正影响 framing 或实现**的文献。广度扫描不进这里，进 `z-doc/README-CN.md` 或 paper 的 related section。
4. 若某篇文献后来被证明不再相关（e.g. 实证反驳），在对应条目加 `**Note (YYYY-MM-DD)**: no longer cited because ...`，**不要删**，留 audit trail。
