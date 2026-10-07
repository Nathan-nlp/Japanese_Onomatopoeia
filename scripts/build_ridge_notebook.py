#!/usr/bin/env python3
"""Build the bilingual Jupyter notebook for the phonological Ridge baseline."""

import json
from pathlib import Path


OUTPUT = Path("notebooks/02_phonological_ridge_baseline.ipynb")
_cell_counter = 0


def next_cell_id() -> str:
    global _cell_counter
    _cell_counter += 1
    return f"ridge-{_cell_counter:02d}"


def code(source: str):
    return {
        "cell_type": "code",
        "id": next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def markdown(source: str):
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    },
    "cells": [],
}

nb["cells"] = [
    markdown(
        """
# 音韵特征Ridge基准 / Phonological Ridge Baseline

本Notebook使用普通词训练音韵—感官评分映射，再在未见过的拟声拟态词上测试。它比较0、16、32个拟声拟态词支持样本，并同时计算平均值基准。

This notebook learns a phonology-to-sensory-rating mapping from ordinary Japanese words and evaluates it on held-out ideophones. It compares 0, 16, and 32 labeled ideophone support items.

**注意：** 三折使用自动生成的词族候选，正式论文分析前需要人工确认。
"""
    ),
    code(
        """
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 100)
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

PROJECT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "japanese_sensory_modeling.csv"
COLAB_DATA_PATH = Path("/content/japanese_sensory_modeling.csv")
DATA_PATH = PROJECT_DATA_PATH if PROJECT_DATA_PATH.exists() else COLAB_DATA_PATH

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Data file not found. In Colab, upload japanese_sensory_modeling.csv "
        "to the session storage (/content), then run this cell again."
    )

print(f"Using data file: {DATA_PATH}")
DATA_PATH
"""
    ),
    markdown(
        """
## 1. 读取并检查数据 / Load and validate the data

主文件保留510个词。正式分析只筛选 `use_in_primary_analysis == 1`，因此不会物理删除6个待核对词。
"""
    ),
    code(
        """
data = pd.read_csv(DATA_PATH)

required_counts = {
    "all_words": len(data),
    "primary_words": int(data["use_in_primary_analysis"].sum()),
    "ordinary_primary": int(((data["use_in_primary_analysis"] == 1) & (data["is_ideophone"] == 0)).sum()),
    "ideophone_primary": int(((data["use_in_primary_analysis"] == 1) & (data["is_ideophone"] == 1)).sum()),
    "excluded_pending": int((data["use_in_primary_analysis"] == 0).sum()),
}
required_counts
"""
    ),
    code(
        """
assert required_counts == {
    "all_words": 510,
    "primary_words": 504,
    "ordinary_primary": 396,
    "ideophone_primary": 108,
    "excluded_pending": 6,
}

data.loc[data["use_in_primary_analysis"] == 0,
         ["word_id", "word_japanese", "romanized", "is_ideophone",
          "rating_issue_codes", "primary_exclusion_reason"]]
"""
    ),
    markdown(
        """
## 2. 定义输入和预测目标 / Define features and targets

模型A只使用12个音韵特征。六项感官平均评分是预测目标。
"""
    ),
    code(
        """
FEATURE_COLUMNS = [
    "feature_mora_count",
    "feature_full_reduplication",
    "feature_sokuon_count",
    "feature_moraic_n_count",
    "feature_long_mark_count",
    "feature_voiced_obstruent_mora_count",
    "feature_p_series_mora_count",
    "feature_vowel_a_mora_count",
    "feature_vowel_i_mora_count",
    "feature_vowel_u_mora_count",
    "feature_vowel_e_mora_count",
    "feature_vowel_o_mora_count",
]

LABEL_COLUMNS = [
    "visual_mean",
    "auditory_mean",
    "haptic_mean",
    "olfactory_mean",
    "gustatory_mean",
    "interoceptive_mean",
]

primary = data.loc[data["use_in_primary_analysis"] == 1].copy()
assert primary[FEATURE_COLUMNS + LABEL_COLUMNS].notna().all().all()
assert set(primary.loc[primary["is_ideophone"] == 1, "outer_fold_provisional"].astype(int)) == {1, 2, 3}

primary.groupby(["analysis_role", "outer_fold_provisional"], dropna=False).size()
"""
    ),
    markdown(
        """
## 3. 建立训练集和测试集 / Build train and test sets

- 所有396个普通词始终进入训练集。
- 当前外层折中的拟声拟态词只用于测试。
- 16和32个支持词从另外两折抽取。
- 同一随机种子下，16个词是32个词的子集。
"""
    ),
    code(
        """
SUPPORT_SIZES = [0, 16, 32]
SUPPORT_SEEDS = [13, 29, 47, 71, 101]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def make_split(frame, test_fold, support_size, seed):
    ordinary = frame.loc[frame["is_ideophone"] == 0].copy()
    ideophones = frame.loc[frame["is_ideophone"] == 1].copy()

    test = ideophones.loc[ideophones["outer_fold_provisional"] == test_fold].copy()
    support_pool = ideophones.loc[ideophones["outer_fold_provisional"] != test_fold].copy()
    shuffled = support_pool.sample(frac=1, random_state=seed)
    support = shuffled.head(support_size).copy()
    train = pd.concat([ordinary, support], ignore_index=True)

    assert set(train["word_id"]).isdisjoint(set(test["word_id"]))
    train_target_families = set(support["family_candidate_key"])
    test_families = set(test["family_candidate_key"])
    assert train_target_families.isdisjoint(test_families)
    return train, test, support


example_train, example_test, example_support = make_split(primary, 1, 16, 13)
{
    "train_rows": len(example_train),
    "test_rows": len(example_test),
    "support_rows": len(example_support),
}
"""
    ),
    markdown(
        """
## 4. 训练平均值基准和Ridge / Train the mean and Ridge baselines

标准化和Ridge正则强度选择都只在训练数据内部完成。外层测试折不会参与模型选择。
"""
    ),
    code(
        """
def average_dimension_spearman(y_true, y_pred):
    values = []
    for column_index in range(y_true.shape[1]):
        correlation = spearmanr(y_true[:, column_index], y_pred[:, column_index]).statistic
        if np.isfinite(correlation):
            values.append(float(correlation))
    return float(np.mean(values)) if values else np.nan


def evaluate_predictions(y_true, y_pred):
    metrics = {
        "mae_overall": mean_absolute_error(y_true, y_pred),
        "spearman_mean": average_dimension_spearman(y_true, y_pred),
    }
    for index, label in enumerate(LABEL_COLUMNS):
        metrics[f"mae_{label.removesuffix('_mean')}"] = mean_absolute_error(
            y_true[:, index], y_pred[:, index]
        )
    return metrics


def fit_ridge(train, test):
    train_groups = train["family_candidate_key"].fillna(train["word_id"])
    inner_cv = GroupKFold(n_splits=3)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={"ridge__alpha": ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
    )
    search.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMNS], groups=train_groups)
    predictions = search.predict(test[FEATURE_COLUMNS])
    return predictions, float(search.best_params_["ridge__alpha"])
"""
    ),
    code(
        """
run_rows = []
prediction_rows = []

for support_size in SUPPORT_SIZES:
    seeds = [SUPPORT_SEEDS[0]] if support_size == 0 else SUPPORT_SEEDS
    for seed in seeds:
        for test_fold in [1, 2, 3]:
            train, test, support = make_split(primary, test_fold, support_size, seed)
            y_true = test[LABEL_COLUMNS].to_numpy()

            mean_prediction = np.tile(
                train[LABEL_COLUMNS].mean().to_numpy(),
                (len(test), 1),
            )
            ridge_prediction, best_alpha = fit_ridge(train, test)

            for model_name, prediction, alpha in [
                ("mean_baseline", mean_prediction, np.nan),
                ("ridge_phonology", ridge_prediction, best_alpha),
            ]:
                metrics = evaluate_predictions(y_true, prediction)
                run_rows.append({
                    "model": model_name,
                    "support_size": support_size,
                    "seed": seed,
                    "test_fold": test_fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "best_alpha": alpha,
                    **metrics,
                })

                for row_position, (_, test_row) in enumerate(test.iterrows()):
                    record = {
                        "model": model_name,
                        "support_size": support_size,
                        "seed": seed,
                        "test_fold": test_fold,
                        "word_id": test_row["word_id"],
                        "word_japanese": test_row["word_japanese"],
                    }
                    for label_index, label in enumerate(LABEL_COLUMNS):
                        short = label.removesuffix("_mean")
                        record[f"true_{short}"] = y_true[row_position, label_index]
                        record[f"pred_{short}"] = prediction[row_position, label_index]
                    prediction_rows.append(record)

run_metrics = pd.DataFrame(run_rows)
predictions = pd.DataFrame(prediction_rows)

run_metrics.to_csv(RESULTS_DIR / "ridge_run_metrics.csv", index=False)
predictions.to_csv(RESULTS_DIR / "ridge_predictions.csv", index=False)

run_metrics.head()
"""
    ),
    markdown(
        """
## 5. 汇总结果 / Summarize the results

下面的表格给出每种条件的平均MAE。MAE越低，预测越准确。
"""
    ),
    code(
        """
summary = (
    run_metrics
    .groupby(["model", "support_size"], as_index=False)
    .agg(
        mae_mean=("mae_overall", "mean"),
        mae_sd=("mae_overall", "std"),
        spearman_mean=("spearman_mean", "mean"),
        runs=("mae_overall", "size"),
    )
)
summary["mae_sd"] = summary["mae_sd"].fillna(0)
summary
"""
    ),
    code(
        """
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=run_metrics,
    x="support_size",
    y="mae_overall",
    hue="model",
    marker="o",
    errorbar="sd",
)
plt.xlabel("Number of labeled ideophone support items")
plt.ylabel("Mean absolute error (lower is better)")
plt.title("Phonological Ridge baseline")
plt.xticks(SUPPORT_SIZES)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ridge_learning_curve.png", dpi=200)
plt.show()
"""
    ),
    markdown(
        """
## 6. 如何解释 / Interpretation checklist

1. Ridge是否优于平均值基准？如果没有，音韵特征目前没有显示稳定预测能力。
2. 16样本是否优于0样本？这反映少量拟声拟态词能否帮助模型适应目标类别。
3. 32样本是否继续改善？如果没有，可能需要检查支持词选择、词族划分或模型形式。
4. 哪个感官维度的MAE下降最多？这可以形成后续语言学分析。

这一阶段不能单独证明音韵具有额外价值。完整论文还需要将模型A与仅文本模型B、文本加音韵模型C进行比较。
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved {OUTPUT}")
