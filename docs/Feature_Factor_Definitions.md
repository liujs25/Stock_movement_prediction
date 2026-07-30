# 报告中出现的因子与特征计算说明

本文解释项目报告中已经出现的因子、特征组和辅助字段。公式与训练/推理代码中的 `FeatureBuilder` 保持一致。报告中用 `*` 表示一组同类特征时，本文按统一公式说明该组内各具体特征的计算方式。

## 1. 统一记号

除特别说明外，所有 rolling、diff、shift、EWM 都在同一个 `sym/date` 分组内计算，不跨股票、不跨交易日。推理时输入窗口通常是单个股票、单个交易日的 100 tick 历史。

| 记号 | 含义 |
|---|---|
| `t` | 当前 tick |
| `eps` | 防止除零的小常数，代码中为 `1e-10` |
| `B_i` | `n_bid{i}`，第 `i` 档买价 |
| `A_i` | `n_ask{i}`，第 `i` 档卖价 |
| `BV_i` | `n_bsize{i}`，第 `i` 档买量 |
| `AV_i` | `n_asize{i}`，第 `i` 档卖量 |
| `M` | `n_midprice`，记录中间价 |
| `C` | `n_close`，记录收盘/价格变动字段 |
| `V` | `amount_delta`，当前 tick 成交额或成交额增量 |
| `rolling_mean_w(x)` | 最近 `w` 个 tick 的滚动均值，`min_periods=1` |
| `rolling_std_w(x)` | 最近 `w` 个 tick 的滚动标准差，`min_periods=1` |
| `shift_w(x)` | 同组内向前滞后 `w` 个 tick |
| `diff_w(x)` | `x_t - x_{t-w}` |
| `EWM_span_s(x)` | span 为 `s` 的指数加权均值，`adjust=False` |

所有最终进入模型的数值都会做清洗：无法转成数值的值、NaN、正负无穷统一替换为 0。

## 2. 原始字段与目标标签

| 字段 | 含义 | 计算方式 |
|---|---|---|
| `sym` | 股票编号 | 从原始字段或文件名解析得到 |
| `date` | 交易日编号 | 从原始字段或文件名解析得到 |
| `time` | tick 时间 | 原始字段，后续用于时间特征 |
| `n_close` | 当前价格/收盘相关标准化字段 | 原始字段，作为基础价格特征 |
| `n_midprice` | 当前中间价 | 原始字段，项目没有用买一卖一均值覆盖它 |
| `amount_delta` | 当前 tick 成交额或成交额增量 | 原始字段 |
| `n_bid1` 至 `n_bid5` | 1 至 5 档买价 | 原始五档盘口字段 |
| `n_ask1` 至 `n_ask5` | 1 至 5 档卖价 | 原始五档盘口字段 |
| `n_bsize1` 至 `n_bsize5` | 1 至 5 档买量 | 原始五档盘口字段 |
| `n_asize1` 至 `n_asize5` | 1 至 5 档卖量 | 原始五档盘口字段 |
| `label_5/10/20/40/60` | 未来不同 tick 间隔的三分类方向标签 | 原始标签字段；只作为目标变量，不作为模型输入 |
| `price_diff_raw` | 验证和阈值分析辅助字段 | `shift_-N(M) - M_t`，其中 `N` 来自对应标签，如 `label_20` 使用未来 20 tick |

`label_*` 的类别含义为：0 表示未来下跌，1 表示未来基本不变，2 表示未来上涨。

## 3. 数据质量相关字段

这些项在报告中用于描述数据质量或清洗逻辑，不是主线模型最终输入特征。

| 名称 | 含义 | 判断或计算方式 |
|---|---|---|
| crossed book | 买价高于卖价的盘口异常 | 典型检查为 `n_bid{i} > n_ask{i}` |
| midprice mismatch | 记录中间价与买一卖一均值不一致 | 比较 `n_midprice` 与 `(n_bid1 + n_ask1) / 2` |
| 涨跌停/异常盘口样本 | 训练阶段默认剔除的样本 | `n_ask1 == 0`、`n_bid1 == 0`、`n_close >= 0.095`、`n_close <= -0.095` |

## 4. 价差、中间价与加权价格

### 4.1 价差特征

报告中出现：`spread_1`、`spread_3`、`spread_5`。

对 `i in {1, 3, 5}`：

```text
spread_i = A_i - B_i
```

含义：第 `i` 档卖价与买价的距离。价差越大，通常表示即时流动性越弱或交易成本越高。

### 4.2 多档中间价

报告中出现：`mid_price_1`、`mid_price_3`、`mid_price_5`。

对 `i in {1, 3, 5}`：

```text
mid_price_i = (A_i + B_i) / 2
```

含义：第 `i` 档价格层级上的买卖中点。它补充了原始 `n_midprice`，让模型看到不同盘口深度处的价格结构。

### 4.3 买卖量加权盘口价

报告中出现：`weighted_ab_1`、`weighted_ab_3`。

对 `i in {1, 3}`：

```text
weighted_ab_i = (B_i * AV_i + A_i * BV_i) / (BV_i + AV_i + eps)
```

含义：用对手盘数量加权的盘口价格。若买量更大，该值会更靠近卖价；若卖量更大，该值会更靠近买价。它反映盘口两侧力量对可成交价格的影响。

### 4.4 Micro price

报告中出现：`micro_price`、`micro_price_diff`。

```text
micro_price = (B_1 * AV_1 + A_1 * BV_1) / (BV_1 + AV_1 + eps)
micro_price_diff = micro_price_t - micro_price_{t-1}
```

含义：一档盘口上的加权价格。`micro_price_diff` 描述该加权价格的短期变化。

## 5. 深度、密度与不平衡因子

### 5.1 相对买卖密度

报告中出现：`relative_bid_density_*`、`relative_ask_density_*`。代码实际计算 `i in {1, 3, 5}`。

```text
relative_bid_density_i = BV_i / (BV_i + AV_i + eps)
relative_ask_density_i = AV_i / (BV_i + AV_i + eps)
```

含义：第 `i` 档盘口中买量或卖量占该档总挂单量的比例。

### 5.2 多档买卖量不平衡

报告中出现：`vol1_rel_diff`、`vol3_rel_diff`、`vol5_rel_diff`。

```text
vol1_rel_diff =
    (BV_1 - AV_1) / (BV_1 + AV_1 + eps)

vol3_rel_diff =
    (sum_{i=1..3} BV_i - sum_{i=1..3} AV_i)
    / (sum_{i=1..3} BV_i + sum_{i=1..3} AV_i + eps)

vol5_rel_diff =
    (sum_{i=1..5} BV_i - sum_{i=1..5} AV_i)
    / (sum_{i=1..5} BV_i + sum_{i=1..5} AV_i + eps)
```

含义：买盘和卖盘挂单量的相对差异。正值表示买盘更厚，负值表示卖盘更厚。

### 5.3 总盘口不平衡

报告中出现：`total_imbalance`。

```text
total_bid_size = sum_{i=1..5} BV_i
total_ask_size = sum_{i=1..5} AV_i

total_imbalance =
    (total_bid_size - total_ask_size)
    / (total_bid_size + total_ask_size + eps)
```

含义：五档总买卖量的不平衡程度。

### 5.4 加权总盘口不平衡

报告中出现：`total_imbalance_weighted`。

```text
weighted_bid = sum_{i=1..5}(B_i * BV_i) / (total_bid_size + eps)
weighted_ask = sum_{i=1..5}(A_i * AV_i) / (total_ask_size + eps)

total_imbalance_weighted =
    (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + eps)
```

含义：先用挂单量计算买卖两侧加权价格，再比较两侧加权价格差异。

### 5.5 加权深度不平衡

报告中出现：`weighted_depth_imbalance`。

代码使用权重：

```text
w = [1.0, 0.8, 0.6, 0.4, 0.2]
bid_depth = sum_{i=1..5}(BV_i * w_i)
ask_depth = sum_{i=1..5}(AV_i * w_i)

weighted_depth_imbalance =
    (bid_depth - ask_depth) / (bid_depth + ask_depth + eps)
```

含义：越靠近最优价的档位权重越高，用于描述更贴近成交位置的盘口压力。

### 5.6 深度压力

报告中出现：`depth_pressure`。

```text
depth_pressure = (bid_depth + ask_depth) * weighted_depth_imbalance
```

含义：在加权深度不平衡的基础上乘以总加权深度。它同时考虑不平衡方向和盘口厚度。

### 5.7 盘口曲率

报告中出现：`book_curvature`。

代码先计算：

```text
bid_convexity = (BV_1 + BV_5 - 2 * BV_3) / (BV_3 + eps)
ask_convexity = (AV_1 + AV_5 - 2 * AV_3) / (AV_3 + eps)

book_curvature = (bid_convexity + ask_convexity) / 2
```

含义：衡量挂单量在浅层、中层、深层之间的弯曲形态。它反映盘口深度是否集中在中间档或两端档。

## 6. 差分、滞后与滚动统计

### 6.1 一阶差分

报告中出现：`close_delta`、`bid1_delta`、`ask1_delta`、`midprice_delta`。

```text
close_delta = C_t - C_{t-1}
bid1_delta = B_1,t - B_1,t-1
ask1_delta = A_1,t - A_1,t-1
midprice_delta = M_t - M_{t-1}
```

含义：描述价格和最优买卖价在相邻 tick 间的变化。

### 6.2 10 tick 滚动均值、标准差和相对均值偏离

报告中出现：各价格和挂单量的 rolling mean、rolling std、相对均值偏离。

对 `n_close`：

```text
close_mean = rolling_mean_10(n_close)
close_std = rolling_std_10(n_close)
close_vs_mean = n_close / (close_mean + eps)
```

对 `i in {1, 3, 5}`，买价、卖价、买量、卖量分别计算：

```text
bid_i_mean = rolling_mean_10(B_i)
bid_i_std = rolling_std_10(B_i)
bid_i_vs_mean = B_i / (bid_i_mean + eps)

ask_i_mean = rolling_mean_10(A_i)
ask_i_std = rolling_std_10(A_i)
ask_i_vs_mean = A_i / (ask_i_mean + eps)

bsize_i_mean = rolling_mean_10(BV_i)
bsize_i_std = rolling_std_10(BV_i)
bsize_i_vs_mean = BV_i / (bsize_i_mean + eps)

asize_i_mean = rolling_mean_10(AV_i)
asize_i_std = rolling_std_10(AV_i)
asize_i_vs_mean = AV_i / (asize_i_mean + eps)
```

对 `n_midprice`：

```text
midprice_mean = rolling_mean_10(M)
midprice_std = rolling_std_10(M)
```

对 `mid_price_i`：

```text
mid_price_i_mean = rolling_mean_10(mid_price_i)
mid_price_i_std = rolling_std_10(mid_price_i)
```

含义：均值给出局部基准，标准差给出局部波动，相对均值偏离用于判断当前盘口是否偏离短期正常水平。

### 6.3 滞后特征

报告中出现：`lag_mid_*`、`lag_bid1_*`、`lag_ask1_*`、`lag_bsize1_*`、`lag_asize1_*`。

```text
lag_mid_l = M_{t-l},                 l in {1, 5, 20}
lag_bid1_l = B_1,t-l,               l in {1, 5}
lag_ask1_l = A_1,t-l,               l in {1, 5}
lag_bsize1_l = BV_1,t-l,            l in {1, 5}
lag_asize1_l = AV_1,t-l,            l in {1, 5}
```

含义：直接保留近期历史状态，帮助模型比较当前盘口和历史盘口。

### 6.4 五期中间价均线

报告中出现：`midprice_ma5`。

```text
midprice_ma5 = rolling_mean_5(M)
```

含义：短窗口价格平滑，用于降低单 tick 噪声。

## 7. 动量、波动和技术指标

### 7.1 收益率/变化率 ROC

报告中出现：`roc_1`、`roc_5`、`roc_10`、`roc_30`、`roc_60`、`roc_100`。

对 `w in {1, 5, 10, 30, 60, 100}`：

```text
roc_w = M_t / M_{t-w} - 1
```

历史不足时填 0。

含义：衡量不同历史间隔下的价格变化率。

### 7.2 波动率特征

报告中出现：`volatility_5`、`volatility_10`、`volatility_20`、`volatility_40`、`volatility_60`。

代码使用：

```text
temp_mid = 2 + A_1 + B_1

volatility_p = temp_mid_t / temp_mid_{t-p} - 1
```

其中 `p in {5, 10, 20, 40, 60}`。历史不足时填 0。

含义：用买一卖一构造的短期价格变化，用于描述局部波动。

### 7.3 MACD

报告中出现：`macd_dif`、`macd_dea`、`macd_bar`。

```text
ema12 = EWM_span_12(M)
ema26 = EWM_span_26(M)

macd_dif = ema12 - ema26
macd_dea = EWM_span_9(macd_dif)
macd_bar = macd_dif - macd_dea
```

含义：衡量短期均线和长期均线的差异及其变化。

### 7.4 KDJ

报告中出现：`kdj_k`、`kdj_d`、`kdj_j`。

```text
low_9 = rolling_min_9(B_1)
high_9 = rolling_max_9(A_1)
RSV = 100 * (M - low_9) / (high_9 - low_9 + eps)

kdj_k = EWM_alpha_1/3(RSV)
kdj_d = EWM_alpha_1/3(kdj_k)
kdj_j = 3 * kdj_k - 2 * kdj_d
```

含义：描述当前中间价在最近 9 tick 买一低点和卖一高点区间中的位置。

### 7.5 价格标准分

报告中出现：`price_zscore_*`。代码实际计算窗口 `w in {20, 100, 300}`。

```text
price_zscore_w =
    (M_t - rolling_mean_w(M)) / (rolling_std_w(M) + eps)
```

含义：当前中间价相对过去 `w` 个 tick 的标准化偏离。

### 7.6 价格斜率

报告中出现：`price_slope_*`。代码实际计算窗口 `w in {20, 100, 300}`。

```text
recent_mean = rolling_mean_{w/3}(M)
early_mean = rolling_mean_{w/3}(shift_{2w/3}(M))

price_slope_w = (recent_mean - early_mean) / (2w/3 + eps)
```

含义：用窗口前后两段均值差估计价格趋势斜率。

### 7.7 价格分位位置

报告中出现：`price_percentile_100`。

```text
price_percentile_100 =
    (M_t - rolling_min_100(M))
    / (rolling_max_100(M) - rolling_min_100(M) + eps)
```

含义：当前价格处于最近 100 tick 价格区间中的相对位置。

## 8. 成交额与流量因子

### 8.1 成交额归一化

报告中出现：`amount_normalized`。

```text
amount_normalized = log1p(V / (1 + M))
```

含义：用中间价缩放成交额，并取 `log1p` 降低极端值影响。

### 8.2 成交额标准分

报告中出现：`amount_zscore_*`。代码实际计算窗口 `w in {20, 100, 300}`。

```text
amount_zscore_w =
    (V_t - rolling_mean_w(V)) / (rolling_std_w(V) + eps)
```

含义：当前成交额相对过去 `w` 个 tick 的标准化偏离。

### 8.3 成交额斜率

报告中出现：`amount_slope_*`。代码实际计算窗口 `w in {20, 100, 300}`。

```text
recent_amount_mean = rolling_mean_{w/3}(V)
early_amount_mean = rolling_mean_{w/3}(shift_{2w/3}(V))

amount_slope_w =
    (recent_amount_mean - early_amount_mean) / (2w/3 + eps)
```

含义：估计成交额在窗口内的趋势变化。

### 8.4 成交流量

报告中出现：`volume_flow_5`、`volume_flow_20`、`volume_flow_60`。

对 `p in {5, 20, 60}`：

```text
volume_flow_p = rolling_sum_p(V) / (rolling_mean_100(V) + eps)
```

含义：最近 `p` 个 tick 的成交额总量相对于 100 tick 平均成交额的强弱。

### 8.5 买卖强度

报告中出现：`buy_intensity`、`sell_intensity`。

```text
buy_intensity = V / (AV_1 + eps)
sell_intensity = V / (BV_1 + eps)
```

含义：用成交额相对于卖一量或买一量的比例近似衡量冲击强度。

### 8.6 VPIN 类流量波动

报告中出现：`vpin_5`、`vpin_20`。

对 `p in {5, 20}`：

```text
vpin_p =
    rolling_std_p(abs(V)) / (rolling_mean_p(abs(V)) + eps)
```

含义：成交额绝对值在短窗口内的波动强度。数值越高，说明流量更不稳定。

## 9. 订单流压力因子

### 9.1 单档 OFI

报告中出现：`ofi_1`。

代码对第 `i` 档定义：

```text
prev_B_i = B_i,t-1
prev_BV_i = BV_i,t-1
prev_A_i = A_i,t-1
prev_AV_i = AV_i,t-1

ofi_bid_i =
    BV_i,t                         if B_i,t > prev_B_i
    BV_i,t - prev_BV_i             if B_i,t == prev_B_i
    -prev_BV_i                     if B_i,t < prev_B_i

ofi_ask_i =
    AV_i,t                         if A_i,t < prev_A_i
    AV_i,t - prev_AV_i             if A_i,t == prev_A_i
    -prev_AV_i                     if A_i,t > prev_A_i

ofi_i = ofi_bid_i - ofi_ask_i
```

`ofi_1` 即第 1 档 OFI。

含义：用买卖价和挂单量的变化估计订单流方向压力。买价上移或买量增加通常增强买方压力；卖价下移或卖量增加通常增强卖方压力。

### 9.2 三档平均 OFI

报告中出现：`ofi_avg_3`。

```text
ofi_avg_3 = (ofi_1 + ofi_2 + ofi_3) / 3
```

含义：平均前 3 档订单流压力，降低单一档位噪声。

### 9.3 OFI 短期累积

报告中出现：`ofi_1_rolling_5`。

```text
ofi_1_rolling_5 = rolling_sum_5(ofi_1)
```

含义：最近 5 tick 的一档订单流压力累计。

### 9.4 OFI 指数均值

报告中出现：`ofi_ema_5`、`ofi_ema_10`。

```text
ofi_ema_5 = EWM_span_5(ofi_1)
ofi_ema_10 = EWM_span_10(ofi_1)
```

含义：平滑后的订单流压力，保留近期变化但降低噪声。

### 9.5 OFI 相对价差

报告中出现：`ofi_spread_ratio`。

```text
ofi_spread_ratio = ofi_1 / (spread_1 + eps)
```

含义：将订单流压力按一档价差缩放。价差越窄，同样 OFI 可能对价格更敏感。

### 9.6 OFI 与价格动量同步

报告中出现：`ofi_momentum_sync`。

```text
ofi_momentum_sync = ofi_1 * midprice_delta
```

含义：当订单流压力方向和中间价变化同向时，该特征为正且绝对值较大；用于刻画盘口压力与价格动量的一致性。

### 9.7 盘口压力

报告中出现：`orderbook_pressure`。

```text
orderbook_pressure =
    (total_bid_size - total_ask_size)
    * (M - B_1)
    / (A_1 - B_1 + eps)
```

含义：将买卖深度差与当前中间价在一档价差中的位置结合，用于描述盘口不平衡对价格的压力。

## 10. 时间特征

### 10.1 日内秒数

报告中出现：`time_seconds`。

```text
time_seconds = hour * 3600 + minute * 60 + second
```

若 `time` 无法解析为 `HH:MM:SS`，则填 0。

### 10.2 日内时间桶

报告中出现：`time_interval`。

```text
time_interval =
    min(int((time_seconds - 34200) / 1800), 7)   if time_seconds >= 34200
    0                                           otherwise
```

其中 `34200` 对应 09:30:00，`1800` 秒对应 30 分钟。

含义：把交易日划分为粗粒度时间段，使模型能够识别不同日内阶段的盘口行为。

## 11. 最终 922 维向量

报告中出现：最近 5 帧完整基础特征、远端摘要、`t0` 至 `t4`、`mid_lag*`、`imb_lag*`。

### 11.1 最近 5 帧完整基础特征

对每个当前样本 `t`，取 182 个基础特征在最近 5 个位置的值：

```text
base_features_t0 = base_features at t
base_features_t1 = base_features at t-1
base_features_t2 = base_features at t-2
base_features_t3 = base_features at t-3
base_features_t4 = base_features at t-4
```

这部分维度为：

```text
182 * 5 = 910
```

训练阶段跳过每组前 4 行，避免这些位置缺少完整历史。

### 11.2 远端摘要特征

对 `lag in {5, 10, 20, 40, 80, 100}`，只取两个摘要量：

```text
mid_lag{lag} = n_midprice_{t-lag}
imb_lag{lag} = total_imbalance_{t-lag}
```

这部分维度为：

```text
6 * 2 = 12
```

### 11.3 最终维度

最终输入向量维度为：

```text
182 * 5 + 6 * 2 = 922
```

含义：最近 5 帧保留完整盘口细节，较远历史只保留价格和盘口不平衡摘要，从而兼顾局部细节与中期状态。

## 12. 报告中提到但没有逐项展开的完整性说明

报告正文按特征家族解释，没有把 182 个基础特征逐个列为清单。本文已经覆盖报告中出现的所有命名特征和通配特征组，并把通配组展开为公式。若需要对训练代码中的 182 个基础特征逐项核对，可直接查看 `FeatureBuilder.base_feature_names` 或训练产物中的 `feature_spec.json`。
