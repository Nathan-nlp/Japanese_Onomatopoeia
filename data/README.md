# Data notes

`processed/japanese_sensory_modeling.csv` 是建模主表。它保留全部510个词，并新增以下字段：

- `use_in_primary_analysis`：1表示进入主要分析，0表示暂不进入。
- `primary_exclusion_reason`：不进入主要分析的原因。
- `analysis_role`：普通词训练源、拟声拟态词目标或待核对词。
- `outer_fold_provisional`：拟声拟态词的暂定三折编号。
- `split_review_status`：词族划分是否仍需人工确认。

正式训练时必须使用 `use_in_primary_analysis == 1` 过滤数据。不要直接删除待核对词。

原始参与者级文件没有复制到仓库，避免混淆原始数据与加工数据。重新生成此文件时使用 `scripts/prepare_modeling_data.py` 并明确提供来源路径。

## 词族审查

`review/ideophone_family_review.xlsx` 保留 108 个主要分析目标词。`Family review` 工作表中的黄色列需要人工确认：

- `review_decision`：确认、拆分、合并或标记为需要讨论。
- `final_family_id`：最终词族编号；拆分或合并时修改。
- `review_notes`：记录依据和不确定之处。

4 个包含多个词的候选词族排在表格最前面。全部 108 行完成审查并写回主表之前，模型输出只能视为暂定结果。
