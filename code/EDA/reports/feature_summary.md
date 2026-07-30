# Feature Engineering Summary

## 1. 特征工程目标
本脚本将 data/raw/ 下的原始股票快照 CSV 文件处理成建模可用的表格特征数据。

## 2. 输入数据
- 原始文件路径: data\raw
- 扫描到的 CSV 文件数量: 1521
- 成功读取文件数: 1521
- 读取失败文件数: 0
- 跳过文件数: 0

## 3. 输出数据
- tabular_features.parquet 路径: data\processed\tabular_features.parquet
- feature_columns.json 路径: data\processed\feature_columns.json
- 最终样本数: 2889900
- 最终总列数: 218
- 最终模型特征数: 207

## 4. 构造的特征类别
- 原始基础特征: n_close 或 close, amount_delta, n_midprice, n_bid1~n_bid5, n_ask1~n_ask5, n_bsize1~n_bsize5, n_asize1~n_asize5
- amount_delta 变换特征: log_amount_delta
- crossed book 特征: is_crossed_book
- midprice 质量检查特征: calc_midprice, midprice_diff, abs_midprice_diff, is_midprice_mismatch
- spread 特征: spread_1~spread_5, spread_pct
- depth 特征: bid_depth, ask_depth, total_depth, depth_diff
- imbalance 特征: imbalance_total, imbalance_1~imbalance_5
- weighted_midprice 特征: weighted_midprice_1
- price diff 特征: mid_diff_1, mid_diff_5, mid_diff_10, mid_diff_20
- rolling 特征说明: 对以下变量计算 rolling mean/std/min/max/change： n_midprice、spread_1、imbalance_total、bid_depth、ask_depth、log_amount_delta；
  另外对 log_amount_delta 还计算 rolling sum（窗口：5/10/20/50/100）

## 5. 清洗策略
- 是否删除 crossed book: 否
- 是否保留 is_crossed_book: 是
- 是否覆盖 n_midprice: 否
- 是否删除 rolling NaN: 是
- 是否补齐缺失文件: 否
- 是否删除最后 60 行: 否

## 6. 标签列
- label_5 (目标变量，不进入 feature_columns.json)
- label_10 (目标变量，不进入 feature_columns.json)
- label_20 (目标变量，不进入 feature_columns.json)
- label_40 (目标变量，不进入 feature_columns.json)
- label_60 (目标变量，不进入 feature_columns.json)

## 7. 数据质量检查
- 原始总行数: 3040479
- Rolling 删除行数: 150579
- 最终样本数: 2889900
- 期望的 rolling 删除后样本数 (原始总行数 - success_files * 99): 2889900
- 实际保存到 parquet 的行数: 2889900
- feature_columns.json 包含 sym_id: 是
- feature_columns.json 包含 session: 是
- feature_columns.json 不包含 date_id: 是
- feature_columns.json 不包含 source_file/date/time/sym/session_name/标签列: 是
- 特征中是否仍有 NaN: 否 (0 个)
- 特征中是否仍有 Inf: 否 (0 个)
- 标签列是否存在: label_5，label_10，label_20，label_40，label_60
- 是否有文件因为行数少于 100 而全部被删除: 否

## 8. 每个文件处理后的行数
- 共处理文件数: 1521
- 详细逐文件结果见: reports\file_feature_row_counts.csv
- 以下为前 10 个文件示例：
- snapshot_sym0_date0_am.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date0_pm.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date10_am.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date10_pm.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date11_am.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date11_pm.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date12_am.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date12_pm.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date13_am.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功
- snapshot_sym0_date13_pm.csv: 原始 1999 行，保留 1900 行，删除 99 行，状态 成功

## 9. 数据泄露检查
- 每个 CSV 文件单独构造特征，不跨股票、不跨日期、不跨 am/pm session。
- 在构造 mid_diff 和 rolling 特征前，已在单个文件内部按 time 升序排序。
- rolling 特征只使用当前 tick 及过去 tick，不使用未来信息。
- mid_diff 特征只使用过去 tick，不使用未来价格。
- label_5、label_10、label_20、label_40、label_60 只作为目标变量，不进入 feature_columns.json。
- date_id 只用于训练/验证/测试集的时间顺序划分，不作为模型输入特征。
- 最后 60 行未被额外删除，因为原始数据已提供标签。

## 10. 缺失字段说明
- 本次特征工程未发现影响主要特征构造的关键字段缺失问题。

## 11. 给建模同学 B 的说明
### 读取数据
```python
import pandas as pd
import json

df = pd.read_parquet('data/processed/tabular_features.parquet')
with open('data/processed/feature_columns.json', 'r', encoding='utf-8') as f:
    feature_columns = json.load(f)

X = df[feature_columns]
y = df['label_5']
```

### 重要提示
- 训练 label_5、label_10、label_20、label_40、label_60 时应分别建模。
- `date_id` 仅用于按时间顺序划分训练/验证/测试集，不作为模型输入特征。
- `sym_id` 和 `session` 当前被保留并作为模型输入特征。
- 训练/验证/测试集必须严格按 `date_id` 的时间顺序划分，不能随机划分。
- feature_columns.json 中不包含标签列和追踪字段（date/time/source_file/sym/session_name/date_id）。
- source_file 和 time 仅用于追踪，不用于训练。