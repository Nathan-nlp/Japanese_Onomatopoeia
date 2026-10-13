# Japanese Onomatopoeia Sensory Prediction

本项目研究日语普通词中学到的感官知识能否迁移到拟声拟态词，并检验音韵特征能否提高零样本与少样本预测性能。

## 当前研究问题

模型先使用普通词学习六项感官评分，再预测没有参与训练的拟声拟态词。实验比较三种拟声拟态词标注量：0、16、32，以及三种输入方式：仅音韵、仅文本、文本加音韵。

当前阶段完成**模型 A：仅音韵特征的 Ridge 回归基准**。仓库同时保留初步版和正式版：

- `02_phonological_ridge_baseline.ipynb`：用于理解流程的初步实验。
- `03_model_a_formal_phonological_ridge.ipynb`：用于论文的正式流程，包含词族隔离、20 个固定种子、分维度调参、置信区间和成对检验。

## 数据范围

- 主文件保留510个词，方便追踪数据质量问题。
- 6个待核对词标记为不进入主要分析，不从主文件删除。
- 主要分析使用504个词：396个普通词和108个拟声拟态词。
- 六项预测目标：视觉、听觉、触觉、嗅觉、味觉、身体内部感觉。
- 三折划分目前基于自动生成的词族候选，状态为暂定；正式论文分析前需要人工确认词族。
- `data/review/ideophone_family_review.xlsx` 是 108 个目标词的人工词族审查表。确认前，正式版 Notebook 会把所有结果标成 `provisional_candidate_families`。

## 项目结构

```text
Japanese_Onomatopoeia/
├── data/
│   ├── README.md
│   ├── processed/
│   │   └── japanese_sensory_modeling.csv
│   └── review/
│       └── ideophone_family_review.xlsx
├── docs/
│   ├── experiment_protocol.md
│   └── model_a_results_provisional.md
├── figures/
├── notebooks/
│   ├── 02_phonological_ridge_baseline.ipynb
│   └── 03_model_a_formal_phonological_ridge.ipynb
├── results/
├── scripts/
│   ├── build_model_a_formal_notebook.py
│   ├── build_ridge_notebook.py
│   └── prepare_modeling_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 运行方法

安装依赖后，在项目根目录运行：

```bash
jupyter lab
```

先打开 `data/review/ideophone_family_review.xlsx` 检查词族。需要理解旧流程时打开 Notebook 02；运行论文版实验时打开 Notebook 03 并依次运行全部单元格。

正式版会生成：

- 逐词预测、逐次运行指标、每个感官维度的最优 alpha 和标准化系数。
- 词族层级 bootstrap 置信区间、成对置换检验和 FDR 校正。
- 固定的少样本支持词清单和普通词—拟声拟态词特征分布检查。
- 学习曲线、分感官误差、系数热图和特征分布变化图。

主要输出使用 `model_a_` 前缀，保存在 `results/` 和 `figures/`。

当前完整运行的暂定结果和谨慎解释见 `docs/model_a_results_provisional.md`。

若只想确认代码能否运行，可在终端设置 `MODEL_A_FAST=1` 后执行 Notebook。论文结果必须使用默认完整模式，不能使用快速检查结果。

## 复现数据准备

整理后的原始主表不直接修改。需要重新生成建模文件时运行：

```bash
python scripts/prepare_modeling_data.py --input "/absolute/path/japanese_sensory_merged.csv"
```

脚本会重新添加主要分析标记，并生成暂定三折划分。原始参与者级数据不要提交到GitHub。
