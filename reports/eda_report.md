# 基于股票快照的价格移动方向预测：EDA 分析报告

## 1. EDA 目标

本次探索性数据分析（Exploratory Data Analysis, EDA）的目标是对股票快照订单簿数据进行系统检查，为后续的数据清洗、特征工程和建模提供依据。重点关注以下问题：

1. 数据文件是否完整，是否存在读取失败、空文件或缺失交易组合；
2. 字段结构是否符合项目说明，关键变量是否可用；
3. 数据中是否存在缺失值、重复值、无穷值等基础质量问题；
4. 订单簿价格和挂单量是否存在异常；
5. `n_midprice` 是否与买一价、卖一价规则基本一致；
6. `label_5`、`label_10`、`label_20`、`label_40`、`label_60` 是否对应未来价格移动方向；
7. 标签类别是否均衡，后续建模是否需要类别权重或采样策略。

---

## 2. 项目任务与数据背景

本项目的任务是基于股票订单簿快照数据，预测未来若干 tick 后中间价的移动方向。预测窗口包括：

- 未来 5 tick；
- 未来 10 tick；
- 未来 20 tick；
- 未来 40 tick；
- 未来 60 tick。

目标变量为三分类标签：

| 标签值 | 含义 |
|---:|---|
| 0 | 下跌 |
| 1 | 不变 |
| 2 | 上涨 |

实际数据中的标签字段为：

```text
label_5, label_10, label_20, label_40, label_60
```

文件命名格式为：

```text
snapshot_sym<xx>_date<yy>_am.csv
snapshot_sym<xx>_date<yy>_pm.csv
```

每个文件表示某一只股票、某一个交易日、某一个交易时段的行情快照数据。

---

## 3. 数据概览

本次 EDA 共读取行情文件 1521 个，全部读取成功。合并后数据规模如下：

| 指标 | 数值 |
|---|---:|
| 总文件数 | 1521 |
| 成功读取文件数 | 1521 |
| 读取失败文件数 | 0 |
| 空文件数 | 0 |
| 总行数 | 3,040,479 |
| 总列数 | 35 |

从基础读取结果看，所有已存在文件均能正常读取，没有发现空文件和读取失败文件，说明原始文件可读性较好。

---

## 4. 字段说明

实际数据中的关键字段包括：

| 字段类型 | 字段 |
|---|---|
| 标识字段 | `date`, `time`, `sym` |
| 价格字段 | `n_close`, `n_midprice`, `n_bid1`–`n_bid5`, `n_ask1`–`n_ask5` |
| 数量字段 | `n_bsize1`–`n_bsize5`, `n_asize1`–`n_asize5` |
| 成交字段 | `amount_delta` |
| 标签字段 | `label_5`, `label_10`, `label_20`, `label_40`, `label_60` |
| EDA 辅助字段 | `source_file`, `sym_id`, `date_id`, `session` 等 |

需要注意，实际数据中使用的是 `n_close`，不是 `close`。因此后续代码和报告均应以实际字段名 `n_close` 为准。

---

## 5. 文件完整性检查

理论上，数据包含：

```text
10 只股票 × 79 个交易日 × 2 个交易时段 = 1580 个文件组合
```

实际文件完整性检查结果如下：

| 指标 | 数值 |
|---|---:|
| 理论文件组合数 | 1580 |
| 实际有效组合数 | 1521 |
| 实际文件数 | 1521 |
| 额外异常组合数 | 0 |
| 缺失组合数 | 59 |

结果说明：理论上应有 1580 个股票-日期-session 组合，实际有效组合为 1521 个，缺失 59 个组合，未发现额外异常组合。缺失详情已保存至：

```text
reports/missing_files.csv
```

该文件可用于后续确认缺失组合是否来自停牌、无交易、数据缺失或文件未提供。

---

## 6. 基础数据质量检查

基础数据质量检查结果如下：

| 检查项 | 结果 |
|---|---:|
| 缺失值 | 未发现明显缺失 |
| 重复行数 | 0 |
| Inf / -Inf 值 | 未发现 |
| 读取失败文件 | 0 |
| 空文件 | 0 |

整体来看，基础数据质量较好，可以进入进一步的订单簿结构检查和标签分析。

---

## 7. 订单簿异常检查

订单簿价格关系检查中，发现部分 crossed book 或档位异常情况。

### 7.1 Bid-Ask 交叉异常

| 异常项 | 数量 | 比例 |
|---|---:|---:|
| `n_bid1 > n_ask1` | 30,853 | 1.01% |
| `n_bid2 > n_ask2` | 32,468 | 1.07% |
| `n_bid3 > n_ask3` | 33,806 | 1.11% |
| `n_bid4 > n_ask4` | 34,696 | 1.14% |
| `n_bid5 > n_ask5` | 35,674 | 1.17% |

其中最重要的是 `n_bid1 > n_ask1`，约占全部样本的 1.01%。正常订单簿中通常应满足 `bid1 <= ask1`，因此这部分样本可视为 crossed book 或盘口异常记录。

### 7.2 Bid 档位内部顺序异常

| 异常项 | 数量 | 比例 |
|---|---:|---:|
| `n_bid2 > n_bid1` | 308 | 0.01% |
| `n_bid3 > n_bid2` | 170 | 0.01% |
| `n_bid4 > n_bid3` | 88 | 约 0.00% |
| `n_bid5 > n_bid4` | 54 | 约 0.00% |

Bid 档位内部错序比例很低，说明买盘档位整体结构较正常。

### 7.3 Ask 档位内部顺序异常

| 异常项 | 数量 | 比例 |
|---|---:|---:|
| `n_ask2 < n_ask1` | 1,307 | 0.04% |
| `n_ask3 < n_ask2` | 1,168 | 0.04% |
| `n_ask4 < n_ask3` | 802 | 0.03% |
| `n_ask5 < n_ask4` | 924 | 0.03% |

Ask 档位内部错序比例也较低。

### 7.4 挂单量异常

所有 size 字段均未发现负数：

```text
n_bsize1 ~ n_bsize5, n_asize1 ~ n_asize5 的负数比例均为 0
```

这说明挂单量字段没有明显非法负值。

### 7.5 订单簿异常处理建议

对于 `n_bid1 > n_ask1` 的样本，不建议在第一版建模前直接全部删除。更稳妥的策略是构造异常标记特征：

```text
is_crossed_book = 1 if n_bid1 > n_ask1 else 0
```

后续可以比较“保留异常标记”和“过滤异常样本”两种策略的验证集表现。

---

## 8. Midprice 规则检查

根据数据说明，若买一价和卖一价均不为 0，则理论中间价应为：

```text
calc_midprice = (n_bid1 + n_ask1) / 2
```

如果一方为 0，则取非零价格；如果两方都为 0，则置为缺失。将该计算结果与数据自带的 `n_midprice` 对比后得到：

| 指标 | 数值 |
|---|---:|
| `midprice_mismatch_count` | 166,530 |
| `midprice_mismatch_ratio` | 5.48% |
| `max_abs_midprice_diff` | 0.0054545455 |
| `mean_abs_midprice_diff` | 0.0000727803 |

结果说明：约 5.48% 的样本中，数据自带 `n_midprice` 与按 `n_bid1`、`n_ask1` 重新计算的中间价存在差异。但平均绝对差异较小，说明大多数不一致可能是轻微差异、数据源计算规则差异、浮点精度或特殊盘口状态导致。

后续建模建议优先使用数据自带的 `n_midprice`，不要直接用重新计算值覆盖原字段。可以额外构造：

```text
midprice_diff = n_midprice - calc_midprice
```

作为数据质量或盘口异常相关特征。

---

## 9. 标签方向检查

由于项目说明中曾出现“当前 tick 相对于 N tick 之前”和“未来 N tick 相对于当前 tick”两种容易混淆的表述，因此本次 EDA 对标签方向进行了验证。

对每个文件单独检查，不跨文件计算，分别比较：

```text
未来方向: n_midprice[t+N] - n_midprice[t]
过去方向: n_midprice[t] - n_midprice[t-N]
```

检查结果如下：

| 标签 | Future 匹配率 | Past 匹配率 | 更符合方向 |
|---|---:|---:|---|
| `label_5` | 1.0000 | 0.6301 | future |
| `label_10` | 1.0000 | 0.5554 | future |
| `label_20` | 1.0000 | 0.5837 | future |
| `label_40` | 1.0000 | 0.4504 | future |
| `label_60` | 1.0000 | 0.3948 | future |

结论：五个标签均完全符合未来方向定义。因此，后续建模目标可以明确为：

```text
使用当前及过去若干 tick 的订单簿快照特征，预测未来 N tick 后中间价的移动方向。
```

---

## 10. 标签分布分析

标签分布如下：

| 标签 | 下跌 0 | 不变 1 | 上涨 2 |
|---|---:|---:|---:|
| `label_5` | 460,804 / 15.16% | 2,122,348 / 69.80% | 457,327 / 15.04% |
| `label_10` | 647,453 / 21.29% | 1,762,358 / 57.96% | 630,668 / 20.74% |
| `label_20` | 520,965 / 17.13% | 2,006,167 / 65.98% | 513,347 / 16.88% |
| `label_40` | 741,438 / 24.39% | 1,586,710 / 52.19% | 712,331 / 23.43% |
| `label_60` | 857,580 / 28.21% | 1,361,229 / 44.77% | 821,670 / 27.02% |

### 10.1 标签分布结论

1. `label_5` 中不变类占比最高，达到 69.80%，类别不均衡最明显；
2. `label_10` 的不变类占比下降到 57.96%，上涨和下跌样本增加；
3. `label_20` 的不变类占比回升到 65.98%，主要原因可能是 N=20 时标签阈值从 0.0005 提高到 0.001；
4. `label_40` 和 `label_60` 的类别分布更均衡，尤其 `label_60` 中上涨、下跌、不变三类比例相对接近；
5. 后续训练时不能只看 accuracy，应重点关注下跌类 0 和上涨类 2 的 precision、recall 和 F0.5。

---

## 11. 可视化结果说明

已生成以下图表：

| 图表 | 文件路径 | 说明 |
|---|---|---|
| `label_5_distribution.png` | `reports/figures/label_5_distribution.png` | 未来 5 tick 标签分布 |
| `label_10_distribution.png` | `reports/figures/label_10_distribution.png` | 未来 10 tick 标签分布 |
| `label_20_distribution.png` | `reports/figures/label_20_distribution.png` | 未来 20 tick 标签分布 |
| `label_40_distribution.png` | `reports/figures/label_40_distribution.png` | 未来 40 tick 标签分布 |
| `label_60_distribution.png` | `reports/figures/label_60_distribution.png` | 未来 60 tick 标签分布 |
| `samples_per_sym.png` | `reports/figures/samples_per_sym.png` | 各股票样本量分布 |
| `samples_per_date.png` | `reports/figures/samples_per_date.png` | 各交易日样本量分布 |
| `samples_per_session.png` | `reports/figures/samples_per_session.png` | 上午/下午样本量分布 |
| `n_midprice_hist.png` | `reports/figures/n_midprice_hist.png` | 中间价分布 |
| `amount_delta_hist.png` | `reports/figures/amount_delta_hist.png` | 成交金额变化分布 |
| `log_amount_delta_hist.png` | `reports/figures/log_amount_delta_hist.png` | 对数变换后的成交金额变化分布 |
| `spread_hist.png` | `reports/figures/spread_hist.png` | bid-ask spread 原始分布 |
| `spread_clipped_hist.png` | `reports/figures/spread_clipped_hist.png` | 1%–99% 截尾后的 spread 分布 |
| `correlation_heatmap.png` | `reports/figures/correlation_heatmap.png` | 主要数值字段相关性热力图 |

---

## 12. 主要结论

本次 EDA 得到以下结论：

1. 数据整体可用。共读取 1521 个 CSV 文件，全部读取成功，无空文件、无读取失败文件。合并后共有 3,040,479 条样本和 35 个字段。

2. 基础数据质量较好。未发现明显缺失值、重复行或无穷值。

3. 文件组合存在缺失。理论上应有 1580 个股票-日期-session 组合，实际有效组合为 1521 个，缺失 59 个组合。后续划分训练集和验证集时应注意这些缺失交易组合。

4. 订单簿整体结构基本合理，但存在少量异常。约 1.01% 的样本存在 `n_bid1 > n_ask1`，即 crossed book 现象；bid/ask 档位内部错序比例较低；size 字段无负数。

5. `n_midprice` 与按 `n_bid1`、`n_ask1` 规则计算的中间价存在约 5.48% 的不一致，但平均绝对差异较小。后续建议优先使用原始 `n_midprice`，而不是直接覆盖。

6. 标签方向已经确认。`label_5`、`label_10`、`label_20`、`label_40`、`label_60` 均完全符合未来方向，即未来 N tick 的中间价相对于当前 tick 的移动方向。

7. 标签存在类别不均衡。尤其 `label_5` 和 `label_20` 中不变类占比较高，后续模型不能只依赖 accuracy，应重点关注上涨和下跌类别的表现。

---

## 13. 后续建议

进入特征工程和建模阶段前，建议采取以下策略：

1. **训练/验证划分**  
   按 `date_id` 做时间序列划分，不建议随机划分样本，避免未来信息泄露。

2. **midprice 处理**  
   优先使用原始 `n_midprice`。如果需要使用重算中间价，应先确认其与标签生成逻辑的一致性。

3. **异常订单簿处理**  
   不建议第一版直接删除 crossed book 样本。可先构造 `is_crossed_book`、`spread`、`abs_spread` 等特征，再比较过滤和不过滤的模型效果。

4. **amount_delta 处理**  
   由于 `amount_delta` 存在明显右偏和极端值，建议使用 `log1p(amount_delta)` 或分位数截尾处理。

5. **类别不均衡处理**  
   训练分类模型时可使用 `class_weight="balanced"`、样本权重，或对上涨/下跌样本进行适度重采样。

6. **评价指标**  
   不应只看 accuracy。建议重点使用 precision、recall、F0.5，尤其关注标签 0 和标签 2 的预测效果。

7. **特征工程方向**  
   后续可构造 spread、order imbalance、bid/ask depth、rolling return、rolling volatility、过去 100 tick 的统计特征等。

---
