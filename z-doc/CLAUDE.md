# CLAUDE.md —— `z-doc/`「对外 presentation」

本目录放所有**给人看**的叙事材料：中英双语 Beamer slides、README-CN（讲解指南）、图表脚本。**不在任何 pipeline 的数据 path 上**，单纯 sink。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)。文档同步规则写在根 [`../PROGRESS.md`](../PROGRESS.md) 顶部。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 承载 `slides.tex` / `slides-en.tex` / `README-CN.md` 三份同步叙事 | 重新跑实验 / 产数字（属阶段 ①–⑤） |
| 承载 `figures/*.py` 从上游 artefact 生成 PDF 图 | 引入新的数据源（图必须 trace 回 artefact） |
| 编译 `slides.pdf` / `slides-en.pdf` 提交入 git | 只管 `.tex` 不管 `.pdf`（两者必须一致） |

## 文件清单

| 文件 | 内容 |
|---|---|
| `slides.tex` / `slides.pdf` | **中文**版 Beamer，含 `\usepackage{ctex}`（中文字体依赖） |
| `slides-en.tex` / `slides-en.pdf` | **英文**版，无 ctex |
| `README-CN.md` | 对外讲解指南（数字、论文定位、与 slides 同步） |
| `figures/make_figures.py` | 从 `5.audit/regression-predictor/audit/` 读 CSV 生成 fig_three_layer_decay / fig_per_forget_profile / fig_audit_scatter |
| `figures/make_fig1_hero.py` | 生成 `fig1_hero.pdf`（一图看懂帧用） |
| `figures/make_fig2_intro_storyboard.py` | 生成 `fig2_intro_storyboard.pdf` |
| `cite/` | bibtex 草稿仓库（paper 阶段消费）|

## 编译方式（决定性信息）

### 首选：tectonic（本仓库默认）

**为什么**：single binary、按需从 CTAN 拉缺失 package、不碰系统 TeX、可在不同机器一致复现。装过一次 `~/bin/tectonic` 即可用。

```bash
cd z-doc/
~/bin/tectonic slides-en.tex     # 英文版（无中文依赖，最省事）
~/bin/tectonic slides.tex        # 中文版（首次编译 tectonic 会拉 ctex + 字体，~1 min）
```

**特点**：
- 一次调用自动完成多次 LaTeX pass（不需要人工 xelatex 跑两次）
- 缺包自动下载（不会像系统 TeX 那样 "package not found"）
- 首跑会拉 CM 字体 + metropolis + ctex（中文版），~1–5 min；之后 <30 s

### 备选：系统 xelatex（若 tectonic 不可用）

```bash
xelatex -interaction=nonstopmode slides-en.tex
xelatex -interaction=nonstopmode slides-en.tex   # 双跑保证 TOC 引用
```

依赖：`apt install texlive-xetex texlive-lang-chinese texlive-fonts-extra`（中文版需要 `texlive-lang-chinese`）。注意**跨机一致性不如 tectonic**（系统包版本漂移可能让同一 .tex 渲染略有差异）。

### Overleaf（远端备选）

Overleaf 编辑器选 **XeLaTeX**。字体和 package 托管在云端，无需本地环境。PDF 要手动下回本地再 commit。

## 对 Claude 的具体要求

1. **改完 .tex 必须重编 .pdf 并一起 commit**。`.tex` 和 `.pdf` 不一致会让 fresh clone 看到的 PDF 与叙事脱节。
2. **三份文档同步规则**（根 [`../PROGRESS.md`](../PROGRESS.md) 顶部）：`slides.tex` / `slides-en.tex` / `README-CN.md` 任何内容 / 数字 / 叙述结构改动，三份都要镜像更新。slides-en 的语言可变、论证骨架和关键数字必须一致。
3. **数字更新触发重生成图**：`5.audit/.../audit/*.csv` 或 `3.inference/extract-ppl/corruption_summary.json` 变了之后，`figures/make_figures.py` 要重跑生成新 PDF，然后 slides 重编。
4. **英文版先编**：`slides-en.tex` 没有中文字体依赖，编译最快最稳；用它先验证叙事改动是否语法正确，再编 `slides.tex`。
5. **overfull 可接受阈值**：metropolis theme 对 `\vbox` 最多溢出 ~20 pt 视觉不明显（被页脚进度条遮盖）；> 50 pt 就会顶出框外看到半行，需要 trim。
6. **不要改 `figures/*.pdf` 直接**：它们是 `make_*.py` 的产物，改源脚本重跑。

## 已踩过的坑（留档）

1. **本机无 xelatex 但有 tectonic**（2026-04-24 发现）：换机后 `xelatex` 命令不在 PATH 里，但 `~/bin/tectonic` 已装。不要白费力气试装 TeX Live 全家桶；**tectonic 够用**。

2. **`ctex` 包只对中文版需要**：`slides-en.tex` 刻意**不** `\usepackage{ctex}`，方便在英文-only 环境编译。修改时不要把两版的 preamble 强同步 —— 差异是有意的。

3. **`figures/` 产物带时间戳 / 数字的文件覆盖要小心**：`fig1_hero.pdf` / `fig2_intro_storyboard.pdf` / `fig_audit_scatter.pdf` 等在 n=50 → n=100 换配置时会**静默覆盖**；要追溯某版数字对应的图，用 `git show <sha>:z-doc/figures/fig_*.pdf` 回查。

4. **`make_figures.py` 硬编码 `5.audit/.../audit/*.csv` 路径**：如果目录重构（历史上已发生 `4.regression-predictor` → `5.audit/regression-predictor`），脚本要同步改路径，否则图拿到**旧 artefact** 就 silently 过期。

5. **`slides.tex` 的 `\institute{...}` 字段**：写实验环境（base model + unlearner + dataset），数字或 base model 改动时要同步。它会显示在 title page。

## Smoke test（~30 s）

```bash
cd z-doc
~/bin/tectonic slides-en.tex > /tmp/tectonic.log 2>&1
pdfinfo slides-en.pdf | grep Pages   # 期望 33 页（n=10 TOFU-aligned 版本）
```

若 `Pages: 33` 且无 `error:` 输出，编译成功。Overfull warnings 正常（见对 Claude 要求 §5）。
