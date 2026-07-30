# EDA Summary

## 1. EDA 目标
- 了解数据文件、字段、数据质量、订单簿异常与标签分布，为后续建模提供依据。

## 2. 数据概览
- 总文件数: 1521
- 成功读取文件数: 1521
- 失败文件数: 0
- 总行数: 3040479
- 总列数: 35

## 3. 字段说明
- 参考预期字段列表（部分字段可能缺失或额外存在，请见下文字段检查结果）。
- 关键字段：date, time, sym, n_close, amount_delta, n_midprice, n_bid*, n_ask*, n_bsize*, n_asize*, label_*. 

## 4. 文件检查结果
- 读取失败文件: []
- 空文件: []

## 4.1 文件完整性
- 理论文件数: 1580
- 实际有效组合数: 1521
- 实际文件数: 1521
- 额外异常组合数: 0
- 缺失组合数: 59
- missing_files.csv 保存路径: D:\stock_movement_prediction\reports\missing_files.csv
- unexpected_files.csv 保存路径: D:\stock_movement_prediction\reports\unexpected_files.csv
- 关系说明: 理论组合数 = 实际有效组合数 + 缺失组合数。实际文件数可能大于有效组合数，说明存在异常文件名或额外组合。

## 5. 数据质量检查
- 缺失值概览（部分）： {}
- 重复行数: 0
- Inf 值计数（部分）: {}

## 6. 订单簿异常检查
- 订单簿异常统计（部分）: {'n_bid1_gt_n_ask1_count': 30853, 'n_bid1_gt_n_ask1_ratio': 0.0101, 'n_bid2_gt_n_ask2_count': 32468, 'n_bid2_gt_n_ask2_ratio': 0.0107, 'n_bid3_gt_n_ask3_count': 33806, 'n_bid3_gt_n_ask3_ratio': 0.0111, 'n_bid4_gt_n_ask4_count': 34696, 'n_bid4_gt_n_ask4_ratio': 0.0114, 'n_bid5_gt_n_ask5_count': 35674, 'n_bid5_gt_n_ask5_ratio': 0.0117, 'n_bid2_gt_n_bid1_count': 308, 'n_bid2_gt_n_bid1_ratio': 0.0001, 'n_bid3_gt_n_bid2_count': 170, 'n_bid3_gt_n_bid2_ratio': 0.0001, 'n_bid4_gt_n_bid3_count': 88, 'n_bid4_gt_n_bid3_ratio': 0.0, 'n_bid5_gt_n_bid4_count': 54, 'n_bid5_gt_n_bid4_ratio': 0.0, 'n_ask2_lt_n_ask1_count': 1307, 'n_ask2_lt_n_ask1_ratio': 0.0004, 'n_ask3_lt_n_ask2_count': 1168, 'n_ask3_lt_n_ask2_ratio': 0.0004, 'n_ask4_lt_n_ask3_count': 802, 'n_ask4_lt_n_ask3_ratio': 0.0003, 'n_ask5_lt_n_ask4_count': 924, 'n_ask5_lt_n_ask4_ratio': 0.0003, 'size_negative_counts': {'n_bsize1': 0, 'n_asize1': 0, 'n_bsize2': 0, 'n_asize2': 0, 'n_bsize3': 0, 'n_asize3': 0, 'n_bsize4': 0, 'n_asize4': 0, 'n_bsize5': 0, 'n_asize5': 0}, 'size_negative_ratio': {'n_bsize1': 0.0, 'n_asize1': 0.0, 'n_bsize2': 0.0, 'n_asize2': 0.0, 'n_bsize3': 0.0, 'n_asize3': 0.0, 'n_bsize4': 0.0, 'n_asize4': 0.0, 'n_bsize5': 0.0, 'n_asize5': 0.0}}

## 6. Midprice 规则检查
- midprice_mismatch_count: 166530
- midprice_mismatch_ratio: 0.0548
- max_abs_midprice_diff: 0.005454545454545501
- mean_abs_midprice_diff: 7.278032488923726e-05
- midprice_check.csv 保存路径: D:\stock_movement_prediction\reports\midprice_check.csv

## 7. 标签方向统计
- label_direction_check.csv 保存路径: D:\stock_movement_prediction\reports\label_direction_check.csv
- label_5: future_ratio=1.0, past_ratio=0.6301 -> better=future
- label_10: future_ratio=1.0, past_ratio=0.5554 -> better=future
- label_20: future_ratio=1.0, past_ratio=0.5837 -> better=future
- label_40: future_ratio=1.0, past_ratio=0.4504 -> better=future
- label_60: future_ratio=1.0, past_ratio=0.3948 -> better=future

## 8. 标签分布分析
### label_5
- 1.0: 2122348 (ratio=0.698)
- 0.0: 460804 (ratio=0.1516)
- 2.0: 457327 (ratio=0.1504)

### label_10
- 1.0: 1762358 (ratio=0.5796)
- 0.0: 647453 (ratio=0.2129)
- 2.0: 630668 (ratio=0.2074)

### label_20
- 1.0: 2006167 (ratio=0.6598)
- 0.0: 520965 (ratio=0.1713)
- 2.0: 513347 (ratio=0.1688)

### label_40
- 1.0: 1586710 (ratio=0.5219)
- 0.0: 741438 (ratio=0.2439)
- 2.0: 712331 (ratio=0.2343)

### label_60
- 1.0: 1361229 (ratio=0.4477)
- 0.0: 857580 (ratio=0.2821)
- 2.0: 821670 (ratio=0.2702)

## 8. 可视化结果说明
- 图表已生成并保存到 reports/figures/，包括标签分布、样本量、n_midprice、amount_delta、log_amount_delta、spread、spread_clipped、缺失比例、相关性热力图等。

## 9. 初步结论
- 共有 59 个理论组合缺失，缺失详情见 missing_files.csv。
- midprice 计算与记录存在较多不一致，差异比率为 0.0548，请进一步校验 n_bid1/n_ask1 和 n_midprice 的来源。
- label_5 更符合未来方向（future_ratio=1.0 > past_ratio=0.6301）。
- label_10 更符合未来方向（future_ratio=1.0 > past_ratio=0.5554）。
- label_20 更符合未来方向（future_ratio=1.0 > past_ratio=0.5837）。
- label_40 更符合未来方向（future_ratio=1.0 > past_ratio=0.4504）。
- label_60 更符合未来方向（future_ratio=1.0 > past_ratio=0.3948）。

## 10. 后续建议
- missing_files.csv 列出缺失组合，可以用来补齐数据或排除缺失日期/品种。
- 若 midprice 误差较大，建议根据 n_bid1/n_ask1 重新计算 midprice 并比对标签生成逻辑。
- 结合标签方向检查结果，优先采用更符合的方向特征进行后续建模。

---
### 生成的图表文件
- label_5: D:\stock_movement_prediction\reports\figures\label_5_distribution.png
- label_10: D:\stock_movement_prediction\reports\figures\label_10_distribution.png
- label_20: D:\stock_movement_prediction\reports\figures\label_20_distribution.png
- label_40: D:\stock_movement_prediction\reports\figures\label_40_distribution.png
- label_60: D:\stock_movement_prediction\reports\figures\label_60_distribution.png
- samples_per_sym: D:\stock_movement_prediction\reports\figures\samples_per_sym.png
- samples_per_date: D:\stock_movement_prediction\reports\figures\samples_per_date.png
- samples_per_session: D:\stock_movement_prediction\reports\figures\samples_per_session.png
- n_midprice_hist: D:\stock_movement_prediction\reports\figures\n_midprice_hist.png
- amount_delta_hist: D:\stock_movement_prediction\reports\figures\amount_delta_hist.png
- log_amount_delta_hist: D:\stock_movement_prediction\reports\figures\log_amount_delta_hist.png
- spread_hist: D:\stock_movement_prediction\reports\figures\spread_hist.png
- spread_clipped_hist: D:\stock_movement_prediction\reports\figures\spread_clipped_hist.png
- correlation_heatmap: D:\stock_movement_prediction\reports\figures\correlation_heatmap.png