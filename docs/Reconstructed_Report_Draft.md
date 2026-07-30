# Stock Movement Prediction Report Draft

> This draft reconstructs the technical structure of the previous report using
> the current project implementation. It should not copy the previous author's
> prose, figures, or leaderboard results. Final metrics should be replaced after
> full training is completed.

## 一、模型选择

本项目目标是基于股票盘口快照数据预测未来多个时间跨度的价格移动方向。预测标签包括 `label_5`、`label_10`、`label_20`、`label_40` 和 `label_60`，每个标签均为三分类任务：

- `0`：未来价格下跌
- `1`：未来价格基本不变
- `2`：未来价格上涨

早期方案可以考虑深度学习模型，例如基于一段历史 tick 序列构造卷积或时序网络。但本任务的数据噪声较高，且有效收益更关注上涨、下跌信号的精度，而不是单纯降低整体分类损失。相比复杂神经网络，树模型有几个实际优势：

- 对特征尺度不敏感，减少标准化和归一化带来的额外风险。
- 能直接处理大量手工构造的盘口、价量和时序统计特征。
- 训练和验证流程更容易解释，便于根据阈值控制交易信号数量。
- 可以利用 XGBoost 的外存训练机制缓解滑动样本膨胀后的内存压力。

因此当前实现采用 XGBoost 多分类模型。每个标签单独训练一个模型，最终推理时输出五个预测结果。

## 二、特征构造与数据处理

当前项目的特征工程由 `Submission_XGBoost/src/feature_builder.py` 实现，并已和旧版本 `Previous version of project/data/data_process.py` 做过一致性检查。当前版本保留旧版核心特征逻辑，同时将训练、推理和提交包构建流程整理为更清晰的工程结构。

### 2.1 原始输入

每个样本来自 100 个 tick 的历史窗口。原始盘口字段主要包括：

- 时间与标识字段：`sym`、`date`、`time`
- 价格字段：`n_close`、`n_midprice`
- 成交字段：`amount_delta`
- 五档买盘：`n_bid1` 至 `n_bid5`，`n_bsize1` 至 `n_bsize5`
- 五档卖盘：`n_ask1` 至 `n_ask5`，`n_asize1` 至 `n_asize5`
- 标签字段：`label_5`、`label_10`、`label_20`、`label_40`、`label_60`

当前实现先为每个 tick 构造基础特征，再通过最近多帧和较远历史摘要拼接成最终向量。旧版与新版的特征一致性检查结果为：

- 基础特征数量：182
- 最终拼接特征数量：922
- 特征名完全一致
- 抽样数值差异最大值为 `0.0`

### 2.2 时间特征

时间字符串会被转换为日内秒数，并进一步离散到交易时段桶中。这样模型既能使用连续时间位置，也能捕捉开盘、午盘、收盘附近不同的交易状态。

主要特征包括：

- `time_seconds`：将 `HH:MM:SS` 转换为秒数。
- `time_interval`：按固定时间间隔划分日内交易阶段。

### 2.3 价量与盘口结构特征

盘口数据的核心信息来自买卖价差、挂单量差异以及多档盘口压力。当前实现构造了以下几类特征：

- 价差：`spread_1`、`spread_3`、`spread_5`
- 多档中间价：`mid_price_1`、`mid_price_3`、`mid_price_5`
- 买卖相对密度：`relative_bid_density_*`、`relative_ask_density_*`
- 加权盘口价格：`weighted_ab_1`、`weighted_ab_3`
- 买卖量不平衡：`vol1_rel_diff`、`vol3_rel_diff`、`vol5_rel_diff`
- 成交额变换：`amount_normalized`
- 挂单量对数变换：`log_bsize*`、`log_asize*`

这些特征的作用是把原始五档盘口压缩成模型更容易利用的供需关系、流动性和短期压力信息。

### 2.4 差分、滚动统计与技术指标

为了描述短期变化，当前实现加入了前后 tick 的价格与盘口差分，例如：

- `close_delta`
- `bid1_delta`
- `ask1_delta`
- `midprice_delta`

为了降低 tick 噪声，还使用同一股票、同一交易日内的滚动统计：

- 价格与盘口均值：`*_mean`
- 波动程度：`*_std`
- 当前值相对均值偏离：`*_vs_mean`

此外，当前特征工程还加入了更长窗口的价格和成交额统计、盘口不平衡、订单流压力以及若干技术指标，例如：

- `macd_dif`、`macd_dea`、`macd_bar`
- `kdj_k`、`kdj_d`、`kdj_j`
- `roc_1`、`roc_5`、`roc_10`、`roc_30`、`roc_60`、`roc_100`
- `price_zscore_*`、`amount_zscore_*`
- `total_imbalance`、`weighted_depth_imbalance`
- `ofi_1`、`ofi_avg_3`、`ofi_ema_5`、`ofi_ema_10`

### 2.5 历史窗口拼接

每个最终训练样本不是单个 tick 的基础特征，而是拼接后的 922 维向量。拼接方式包括：

- 最近 5 帧完整基础特征：`t0` 至 `t4`
- 更远历史的摘要特征：在 `5`、`10`、`20`、`40`、`80`、`100` 个 tick 的间隔上抽取 `n_midprice` 与 `total_imbalance`

这种结构兼顾了高频的近期状态和较长窗口的方向性变化。

### 2.6 异常样本处理

当前训练代码支持过滤涨跌停或盘口异常样本。异常判断主要包括：

- `n_ask1 == 0`
- `n_bid1 == 0`
- `n_close >= 0.095`
- `n_close <= -0.095`

该步骤用于减少无法正常交易或盘口失真的样本对模型的干扰。

### 2.7 训练集与验证集划分

当前 `Submission_XGBoost/src/train.py` 支持两种划分方式：

- `date`：按日期切分，默认使用较早日期训练、较晚日期验证。
- `index`：按文件顺序做 8:2 切分，用于复现旧版训练方式。

在复现旧版内容时，应优先使用 `--split-mode index --test-size 0.2`，因为旧版报告中的实验描述接近按时间顺序的 8:2 切分。

## 三、训练方法

当前训练流程使用 XGBoost 的多分类目标函数：

- `objective`: `multi:softprob`
- `num_class`: `3`
- `eval_metric`: `mlogloss`

每个标签单独训练一个模型。训练时先将 CSV 文件处理为外存缓存，再通过 XGBoost `DataIter` / `ExtMemQuantileDMatrix` 载入训练数据。这样可以避免滑动窗口展开后一次性占用过多内存。

当前新实现相对旧版做了工程化整理：

- 统一配置训练参数和输出目录。
- 保存 `feature_spec.json`，保证推理阶段使用同一套特征顺序。
- 保存 `thresholds.json`，将置信度阈值纳入提交包。
- 支持 CPU 和 GPU 训练。
- 支持构建平台提交包并进行 smoke test。

当前项目记录中的 GPU 全量训练计划使用如下核心设置：

- `num_boost_round`: `2000`
- `early_stopping_rounds`: `100`
- `max_bin`: `256`
- `split_mode`: `index`
- `test_size`: `0.2`
- `device`: `cuda`

## 四、信号控制与阈值策略

本任务并不只追求整体 accuracy。由于 `0` 和 `2` 表示可交易方向，错误的上涨/下跌判断通常比把样本判为 `1` 更重要。因此推理阶段使用置信度阈值控制交易信号：

- 如果模型对 `0` 或 `2` 的最大概率超过阈值，则输出对应方向。
- 如果置信度不足，则输出 `1`，表示不交易或无明显方向。

当前 smoke run 使用的默认阈值为：

- `label_5`: `0.88`
- `label_10`: `0.88`
- `label_20`: `0.88`
- `label_40`: `0.88`
- `label_60`: `0.88`

全量训练完成后，应根据验证集的收益、信号比例和方向精度重新选择每个标签的阈值。

## 五、当前实验状态

当前报告重点采用 `Previous version of project/` 当前 HEAD
`b5a8b00` 的完整复现结果。该复现直接运行旧项目当前版本自己的
`train.py`、`data/data_process.py` 和 `test.py`，不使用
`Reproduction_XGBoost` 作为代理。

当前 HEAD 相比历史 `8855d4b` 版本有以下关键差异：

- 基础特征增加到 182 个，最终拼接后为 922 维。
- 最近完整帧从 3 帧扩展到 5 帧。
- 训练时过滤涨跌停或异常盘口样本。
- 每个股票、日期分组前 4 行样本被跳过。
- XGBoost 参数使用 `max_depth=8`、`subsample=0.6`、`reg_alpha=10.0`。

本次已完整训练并评估 `label_20`。其他标签仍需继续训练，当前提交候选为了先形成可提交版本，只对 `label_20` 使用训练模型，其他标签暂时输出 `Unchanged=1`。

### 5.1 `label_20` 当前版本复现

训练集和验证集由当前旧代码按文件顺序 8:2 切分生成。训练样本经过涨跌停过滤和前 4 行跳过处理。

| 项目 | 数值 |
|---|---:|
| 基础特征数 | 182 |
| 最终特征维度 | 922 |
| 训练样本数 | 2,249,985 |
| 验证样本数 | 582,943 |
| 训练集过滤涨跌停样本 | 176,325 |
| 验证集过滤涨跌停样本 | 25,591 |
| Down 训练样本 | 386,426 |
| Unchanged 训练样本 | 1,476,616 |
| Up 训练样本 | 386,943 |

训练使用 XGBoost `multi:softprob` 三分类目标，`num_boost_round=2000`，
`early_stopping_rounds=100`。最终训练在第 596 轮停止，验证集
`mlogloss` 在第 496 轮附近达到最优：

| iteration | eval mlogloss |
|---:|---:|
| 450 | 0.93606 |
| 480 | 0.93594 |
| 490 | 0.93590 |
| 510 | 0.93589 |
| 550 | 0.93589 |
| 596 | 0.93613 |

### 5.2 阈值结果

当前 `test.py` 使用验证集 price diff 计算本地 PnL proxy。该 PnL 不是平台官方 PnL 公式，只用于本地比较阈值。

| threshold | trades | precision | recall | total pnl | avg pnl |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 355,704 | 0.3569 | 0.5703 | 209.761993 | 0.000590 |
| 0.70 | 54,262 | 0.5606 | 0.1366 | 75.342178 | 0.001388 |
| 0.75 | 33,004 | 0.6042 | 0.0896 | 52.402393 | 0.001588 |
| 0.80 | 17,826 | 0.6501 | 0.0520 | 32.434612 | 0.001820 |
| 0.82 | 13,149 | 0.6700 | 0.0396 | 25.249454 | 0.001920 |
| 0.84 | 9,312 | 0.6891 | 0.0288 | 19.106995 | 0.002052 |
| 0.85 | 7,673 | 0.7021 | 0.0242 | 16.175755 | 0.002108 |
| 0.88 | 3,921 | 0.7348 | 0.0129 | 9.087589 | 0.002318 |
| 0.90 | 2,286 | 0.7638 | 0.0078 | 5.724861 | 0.002504 |

阈值提高后，precision 和 avg pnl 上升，但 recall 和 total pnl 下降。若以接近 `0.04` 的 recall 作为折中目标，`threshold=0.82` 是当前验证集上较合理的选择。

### 5.3 提交候选

已构建一个 `label_20` 阈值为 `0.82` 的提交候选：

```text
Submission_XGBoost/artifacts/submission_previous_current_label20_thr082.zip
```

该提交包为 flat package，包含 `Predictor.py`、当前版本特征处理逻辑、`model_label_20.json` 和阈值配置。smoke test 已通过，能够返回平台要求的 `List[List[int]]`，每行包含五个标签预测。

需要明确的是，该候选包目前只训练并启用了 `label_20` 模型。由于当前 HEAD 其他四个标签尚未完成全量训练，提交包中 `label_5`、`label_10`、`label_40`、`label_60` 暂时输出 `1`。如果要形成最终完整版本，还需要对其余四个标签重复同样的当前版本训练和阈值选择流程。

## 六、历史版本收益与平台观察汇总

本节汇总此前讨论中用户提供的平台观测结果，以及本地复现过程中得到的收益和分数。需要区分两类指标：

- 本地 `total pnl` / `avg pnl` 是根据验证集 price diff 计算的 proxy，并不等同于平台官方收益公式。
- 平台截图或平台页面上的 `single pnl` / `average_pnl(BP)` 是用户提供的外部观测，应作为最终现象记录，但平台没有公开完整平均收益计算细节。
- 本地 `official_score_est` 使用当时讨论过的估计公式：
  `F0.5 * (avg_pnl - 0.0004)^2 * sign(avg_pnl - 0.0004) * 10000`。

当前本地 PnL proxy 的方向逻辑为：

- 预测 `2`：`future_n_midprice - current_n_midprice`
- 预测 `0`：`current_n_midprice - future_n_midprice`
- 预测 `1`：不交易，不计入平均 PnL 分母

因此 raw `avg pnl * 10000` 可以近似对应 BP 展示，但这只是本地解释，不保证和平台完全一致。

### 6.1 本地验证版本对照

| 版本 / 策略 | 标签范围 | 信号数 | total pnl | avg pnl / BP | 说明 |
|---|---:|---:|---:|---:|---|
| score-threshold legacy package | 5 labels | 101,237 | 162.90 | 16.09 BP | 早期较均衡版本，本地验证 precision、收益、信号量较折中 |
| avg-pnl min1pct turbo | 5 labels | 37,250 | 71.97 | 19.32 BP | 本地平均收益更高，但信号更少 |
| pure avg-pnl turbo | 5 labels | 1,812 | 5.92 | 32.66 BP | 单笔平均最高，但交易极少，不适合作为稳定提交 |
| total-pnl turbo | 5 labels | 1,853,493 | 920.59 | 4.97 BP | 累计收益最高，但单笔收益和信号质量较低 |
| profit-weighted training | 5 labels | 149,908 | 214.90 | 14.34 BP | 累计收益高于 score-threshold，但平均收益低于 score-threshold |
| legacy 1225 proxy, `label_20`, threshold `0.82` | label_20 | 16,008 | 30.0630 | 18.78 BP | precision `0.6611`，recall `0.0461` |
| legacy 1225 proxy, `label_20`, threshold `0.88` | label_20 | 5,175 | 11.8195 | 22.84 BP | precision `0.7273`，recall `0.0164`，recall 明显低于旧平台目标 |
| current Previous-Version HEAD, `label_20`, threshold `0.82` | label_20 | 13,149 | 25.249454 | 19.20 BP | precision `0.6700`，recall `0.0396` |
| current Previous-Version HEAD, `label_20`, threshold `0.88` | label_20 | 3,921 | 9.087589 | 23.18 BP | precision `0.7348`，recall `0.0129` |

本地结果的共同现象是：提高阈值通常会提高 precision 和单笔 avg pnl，但会快速降低 recall 和 total pnl。当前 HEAD 在 `threshold=0.82` 时 recall 接近 `0.04`，但 precision 和 avg pnl 仍低于用户曾经提供的旧平台强结果。

`full_gpu_legacy` 中按 `official_score_est` 选择阈值的完整分数如下：

| label | threshold | signals | precision | recall | F0.5 | total pnl | avg pnl BP | official_score_est |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| label_5 | 0.88 | 12,150 | 0.8016 | 0.0516 | 0.2053 | 14.72 | 12.11 | 0.001352 |
| label_10 | 0.82 | 22,065 | 0.7671 | 0.0642 | 0.2404 | 30.09 | 13.64 | 0.002233 |
| label_20 | 0.78 | 23,551 | 0.6325 | 0.0669 | 0.2351 | 40.62 | 17.25 | 0.004127 |
| label_40 | 0.70 | 21,546 | 0.6140 | 0.0430 | 0.1681 | 38.51 | 17.87 | 0.003235 |
| label_60 | 0.65 | 21,925 | 0.5980 | 0.0375 | 0.1500 | 38.96 | 17.77 | 0.002844 |

其他几个本地策略的 selected-threshold score 摘要如下：

| strategy | label | threshold | signals | precision | recall | F0.5 | total pnl | avg pnl BP | official_score_est |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| avg-pnl min1pct | label_5 | 0.90 | 8,101 | 0.8261 | 0.0355 | 0.1514 | 10.65 | 13.15 | 0.001267 |
| avg-pnl min1pct | label_10 | 0.88 | 8,042 | 0.8242 | 0.0251 | 0.1120 | 13.32 | 16.56 | 0.001767 |
| avg-pnl min1pct | label_20 | 0.85 | 7,960 | 0.6971 | 0.0249 | 0.1090 | 16.78 | 21.08 | 0.003179 |
| avg-pnl min1pct | label_40 | 0.78 | 6,070 | 0.6931 | 0.0137 | 0.0634 | 14.31 | 23.58 | 0.002430 |
| avg-pnl min1pct | label_60 | 0.72 | 7,077 | 0.6730 | 0.0136 | 0.0631 | 16.91 | 23.90 | 0.002497 |
| pure avg-pnl | all labels | mixed | 1,812 | high | near zero | very low | 5.92 | 32.66 | low because F0.5 is tiny |
| total-pnl | all labels | 0.00 | 1,853,493 | 0.3573-0.4435 | 0.4822-0.6430 | 0.3860-0.4700 | 920.59 | 4.97 | low because avg pnl is close to threshold |

`profit-weighted training` 的 selected-threshold score proxy 如下：

| label | threshold | signals | precision | recall | F0.5 | total pnl | avg pnl BP | official_score_est proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| label_5 | 0.94 | 12,923 | 0.4880 | 0.0330 | 0.1299 | 16.35 | 12.65 | 0.000972 |
| label_10 | 0.90 | 22,229 | 0.5660 | 0.0480 | 0.1792 | 30.50 | 13.72 | 0.001693 |
| label_20 | 0.85 | 33,351 | 0.4780 | 0.0720 | 0.2246 | 47.22 | 14.16 | 0.002319 |
| label_40 | 0.75 | 40,851 | 0.4990 | 0.0660 | 0.2158 | 60.92 | 14.91 | 0.002569 |
| label_60 | 0.70 | 40,554 | 0.5090 | 0.0590 | 0.2016 | 59.91 | 14.77 | 0.002338 |

当前 HEAD 和 legacy 1225 的 `label_20` 关键阈值 score proxy 为：

| version | threshold | signals | precision | recall | F0.5 | total pnl | avg pnl BP | official_score_est |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy 1225 proxy | 0.82 | 16,008 | 0.6611 | 0.0461 | 0.1802 | 30.0630 | 18.78 | 0.003937 |
| legacy 1225 proxy | 0.88 | 5,175 | 0.7273 | 0.0164 | 0.0752 | 11.8195 | 22.84 | 0.002670 |
| current Previous-Version HEAD | 0.82 | 13,149 | 0.6700 | 0.0396 | 0.1601 | 25.249454 | 19.20 | 0.003700 |
| current Previous-Version HEAD | 0.88 | 3,921 | 0.7348 | 0.0129 | 0.0603 | 9.087589 | 23.18 | 0.002217 |

### 6.2 用户提供的平台观测

用户曾提交 `avg_pnl_min1pct_turbo`，平台只完成了其中三个任务。观测结果如下：

| 平台任务 | precision | recall | F0.5 derived | total pnl | single pnl |
|---|---:|---:|---:|---:|---:|
| label_5 | 0.8247 | 0.0353 | 0.1507 | -0.095 | -0.000019 |
| label_10 | 0.8288 | 0.0263 | 0.1167 | 2.0224 | 0.000387 |
| label_20 | 0.7101 | 0.0259 | 0.1130 | 3.5769 | 0.000709 |

这说明提交包的阈值逻辑确实生效，但平台上的单笔收益显著低于本地验证估计。

用户还提供过一个旧平台强结果，核心目标为 `label_20`：

| task | average_pnl(BP) | precision | recall | f0.5 |
|---|---:|---:|---:|---:|
| label20 | 24.2013 | 0.729054 | 0.0398745 | 0.163584 |

该结果的特征是 precision 高、recall 约 `0.04`、平均收益约 `24.20 BP`。这不是单纯极高阈值下的极稀疏策略，因为当前 HEAD 在 `threshold=0.88` 可以接近 precision 和 avg pnl，但 recall 只有 `0.0129`；在 `threshold=0.82` recall 接近，但 precision 和 avg pnl 不足。

后续一次偏高阈值、偏 PnL 的提交在平台上表现很差：

| 平台任务 | precision | recall | f0.5 | total pnl | single pnl |
|---|---:|---:|---:|---:|---:|
| label_5 | 0.953488 | 0.000345951 | 0.00172724 | -0.269940 | -0.006278 |
| label_10 | 1.000000 | 0.0000850717 | 0.000425212 | -0.111900 | -0.007993 |
| label_20 | 0.921260 | 0.000845461 | 0.00421183 | -0.440893 | -0.003472 |
| label_40 | 0.855491 | 0.000778517 | 0.00387845 | -0.166795 | -0.000964 |
| label_60 | 0.822581 | 0.000710792 | 0.0035417 | 0.123083 | 0.000662 |

这个结果说明，高类别概率或高 precision 不等价于高平台收益。过高阈值会导致信号极少，少量错误或平台收益定义差异会主导最终 PnL。

### 6.3 外部榜单式结果参考

用户还提供过一个外部 XGBoost 风格结果，用于理解平台上高收益策略的形态：

| task | total pnl | avg pnl | precision | recall | F0.5 derived | official_score_est proxy |
|---|---:|---:|---:|---:|---:|---:|
| label_5 | 71.16 | 0.000330 | 0.306 | 0.557 | 0.3363 | -0.000016 |
| label_10 | 106.28 | 0.000462 | 0.358 | 0.501 | 0.3797 | 0.000015 |
| label_20 | 225.37 | 0.001063 | 0.290 | 0.444 | 0.3116 | 0.001370 |
| label_40 | 343.39 | 0.001544 | 0.336 | 0.393 | 0.3460 | 0.004529 |
| label_60 | 358.54 | 0.001572 | 0.355 | 0.376 | 0.3590 | 0.004931 |

该结果和我们当前高阈值路线不同：它 precision 不高，但 recall 很高，整体更像“高覆盖的正期望策略”，而不是只筛少量高置信度交易。

综合来看，旧平台强结果仍未被当前 `Previous version of project/` HEAD 完整复现。可能的不一致来源包括：模型 artifact 与代码版本不匹配、训练/验证或平台切分不同、阈值与后处理逻辑不同、平台数据分布变化，或平台收益公式与本地 proxy 存在差异。

## 七、结果分析框架

最终结果分析建议按标签分别展示。每个标签应包含：

- 类别分布：说明 `1` 类是否占多数。
- 验证集分类报告：关注 `0` 和 `2` 的 precision，而不是只看 overall accuracy。
- 混淆矩阵：观察 `0` 与 `2` 是否互相混淆。
- 阈值表：说明提高阈值后信号数量与信号质量如何变化。
- 收益评估：结合比赛规则或本地 PnL 近似指标说明模型是否有实际价值。

预期上，预测间隔越长，噪声累积越明显，方向预测通常会更困难。因此 `label_5` 和 `label_10` 的可预测性一般会强于 `label_40` 和 `label_60`。这需要用全量训练后的验证集结果确认。

## 八、不足与改进

当前方案仍有几个主要改进方向：

- 参数调优：当前训练参数主要延续旧版逻辑，后续可以系统搜索 `max_depth`、`learning_rate`、`subsample`、`colsample_bytree`、正则项和类别权重。
- 特征筛选：922 维特征中可能存在冗余特征，可通过特征重要性、相关性和消融实验筛选。
- 阈值优化：不同标签可以使用不同阈值，优化目标应贴近最终评测收益，而不是固定使用统一阈值。
- 切分策略：需要比较 `index` 切分与按日期切分，确认验证结果是否稳定。
- 训练规模：当前本地结果只是 smoke run，必须完成全量训练后才能写正式结论。
- 推理效率：提交包需要在平台限制下稳定运行，模型数量、特征构造速度和依赖体积都需要验证。

## 九、可复现命令

旧版兼容的 GPU 全量训练命令：

```bash
cd /mnt/stock_data/stock_movement/Submission_XGBoost
source .venv/bin/activate
python src/train.py \
  --split-mode index \
  --test-size 0.2 \
  --device cuda \
  --nthread -1 \
  --max-bin 256 \
  --num-boost-round 2000 \
  --early-stopping-rounds 100 \
  --batch-size 5000 \
  --output-dir artifacts/full_gpu_legacy \
  --cache-dir /mnt/stock_data/stock_movement/cache/full_gpu_legacy \
  --cleanup-cache
```

构建提交包：

```bash
python src/build_submission.py --output-dir artifacts/full_gpu_legacy
```

本地 smoke test：

```bash
python src/smoke_test_submission.py \
  --data-dir "../EDA/raw data/FBDQA2021A_MMP_Challenge/data" \
  --package-dir artifacts/full_gpu_legacy/submission_package
```

## 十、和旧报告的对应关系

| 旧报告部分 | 当前可复现方式 | 注意事项 |
|---|---|---|
| 模型选取 | 写成 XGBoost 相比深度模型更适合当前工程约束 | 不复制旧作者个人探索叙述 |
| 特征构造 | 使用 `feature_builder.py` 的 182/922 维特征说明 | 可引用一致性检查结果 |
| 数据处理 | 说明异常过滤、外存缓存、8:2 切分 | 切分方式必须写清楚 |
| 训练 | 说明 XGBoost 参数、early stopping、外存训练 | 使用我们实际命令 |
| 降低 recall | 改写为置信度阈值策略 | 使用我们的 `thresholds.json` |
| 结果 | 用全量训练后的真实结果替换 | 不使用旧报告公榜/私榜结果 |
| 分析 | 讨论噪声、长周期预测变难、特征冗余 | 可以保留观点，但需用我们结果支撑 |
