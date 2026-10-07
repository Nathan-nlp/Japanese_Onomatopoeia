# Japanese Onomatopoeia Sensory Prediction

本项目研究日语普通词中学到的感官知识能否迁移到拟声拟态词，并检验音韵特征能否提高零样本与少样本预测性能。

## 当前研究问题

模型先使用普通词学习六项感官评分，再预测没有参与训练的拟声拟态词。实验比较三种拟声拟态词标注量：0、16、32，以及三种输入方式：仅音韵、仅文本、文本加音韵。

当前阶段先完成**模型A：仅音韵特征的Ridge回归基准**。

## 数据范围

- 主文件保留510个词，方便追踪数据质量问题。
- 6个待核对词标记为不进入主要分析，不从主文件删除。
- 主要分析使用504个词：396个普通词和108个拟声拟态词。
- 六项预测目标：视觉、听觉、触觉、嗅觉、味觉、身体内部感觉。
- 三折划分目前基于自动生成的词族候选，状态为暂定；正式论文分析前需要人工确认词族。

## 项目结构

```text
Japanese_Onomatopoeia/
├── data/
│   ├── README.md
│   └── processed/
│       └── japanese_sensory_modeling.csv
├── docs/
│   └── experiment_protocol.md
├── figures/
├── notebooks/
│   └── 02_phonological_ridge_baseline.ipynb
├── results/
├── scripts/
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

打开 `notebooks/02_phonological_ridge_baseline.ipynb`，依次运行全部单元格。Notebook会生成：

- `results/ridge_run_metrics.csv`
- `results/ridge_predictions.csv`
- `figures/ridge_learning_curve.png`

## 复现数据准备

整理后的原始主表不直接修改。需要重新生成建模文件时运行：

```bash
python scripts/prepare_modeling_data.py --input "/absolute/path/japanese_sensory_merged.csv"
```

脚本会重新添加主要分析标记，并生成暂定三折划分。原始参与者级数据不要提交到GitHub。

