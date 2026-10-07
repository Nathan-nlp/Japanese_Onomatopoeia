# Data notes

`processed/japanese_sensory_modeling.csv` 是建模主表。它保留全部510个词，并新增以下字段：

- `use_in_primary_analysis`：1表示进入主要分析，0表示暂不进入。
- `primary_exclusion_reason`：不进入主要分析的原因。
- `analysis_role`：普通词训练源、拟声拟态词目标或待核对词。
- `outer_fold_provisional`：拟声拟态词的暂定三折编号。
- `split_review_status`：词族划分是否仍需人工确认。

正式训练时必须使用 `use_in_primary_analysis == 1` 过滤数据。不要直接删除待核对词。

原始参与者级文件没有复制到仓库，避免混淆原始数据与加工数据。重新生成此文件时使用 `scripts/prepare_modeling_data.py` 并明确提供来源路径。
