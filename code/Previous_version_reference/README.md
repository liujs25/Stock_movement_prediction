# X_Stock 使用说明

本项目用于基于盘口快照数据训练 XGBoost 模型，预测股票价格方向。

标签含义：

- `label_5`
- `label_10`
- `label_20`
- `label_40`
- `label_60`

模型输出三分类结果：

- `0`：下跌
- `1`：基本不变
- `2`：上涨

## 环境安装

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

## 数据准备

数据目录中应放置 CSV 文件，文件名格式类似：

```text
snapshot_sym1_date60_am.csv
snapshot_sym1_date60_pm.csv
```

脚本会匹配：

```text
snapshot_sym*_date*_*.csv
```

CSV 中需要包含训练所需字段，例如：

- `sym`
- `date`
- `time`
- `n_close`
- `n_midprice`
- `amount_delta`
- `n_bid1` 到 `n_bid5`
- `n_ask1` 到 `n_ask5`
- `n_bsize1` 到 `n_bsize5`
- `n_asize1` 到 `n_asize5`
- `label_5`、`label_10`、`label_20`、`label_40`、`label_60`

## 训练模型

训练单个标签：

```powershell
python train.py --data_dir ./your_data --model_save_dir ./Experiments --cache_dir ./cache --label label_5
```

训练全部标签：

```powershell
python train.py --data_dir ./your_data --model_save_dir ./Experiments --cache_dir ./cache --label all
```

常用参数：

- `--data_dir`：CSV 数据目录
- `--model_save_dir`：模型保存目录，默认 `./Experiments`
- `--cache_dir`：缓存目录
- `--label`：训练标签，可选 `label_5`、`label_10`、`label_20`、`label_40`、`label_60`、`all`
- `--window`：历史窗口大小，默认 `100`
- `--test_size`：验证集比例，默认 `0.2`
- `--batch_size`：缓存分块大小，默认 `5000`
- `--use_cache`：使用已有缓存，不重新处理原始 CSV
- `--resume`：从最近 checkpoint 继续训练

使用已有缓存训练：

```powershell
python train.py --data_dir ./your_data --model_save_dir ./Experiments --cache_dir ./cache --label label_5 --use_cache
```

断点续训：

```powershell
python train.py --data_dir ./your_data --model_save_dir ./Experiments --cache_dir ./cache --label label_5 --resume
```

训练完成后，模型会保存为：

```text
./Experiments/model_label_5.pth
```

checkpoint 会保存在：

```text
./Experiments/checkpoints/
```

## 评估模型

```powershell
python test.py --model_path ./Experiments/model_label_5.pth --data_dir ./your_data --cache_dir ./cache --label label_5 --threshold 0.88
```

该脚本会读取验证集缓存，输出不同置信度阈值下的信号比例、准确率、召回率和 PnL 统计。

## 分析特征重要性

```powershell
python analyze_features.py --model_path ./Experiments/model_label_5.pth --data_dir ./your_data
```

输出模型 Top 20 特征重要性，并保存图片到：

```text
./Experiments/feature_importance_analysis.png
```

## 统计理论最大 PnL

```powershell
python count_max_pnl.py --label label_5 --cache_dir ./cache --top_percent 5
```

只分析训练集：

```powershell
python count_max_pnl.py --label label_5 --cache_dir ./cache --train_only
```

只分析验证集：

```powershell
python count_max_pnl.py --label label_5 --cache_dir ./cache --val_only
```

该脚本基于缓存中的真实标签和未来价格变化，统计收益分布及前若干比例交易的 PnL。

## 绘制 K 线和买卖信号

```powershell
python draw_candle.py --data_dir ./your_data --model_path ./Experiments/model_label_5.pth --threshold 0.88 --output_dir ./candle_plots
```

常用参数：

- `--threshold`：信号置信度阈值，默认 `0.88`
- `--interval_seconds`：K 线周期，默认 `30`
- `--output_dir`：图片输出目录，默认 `./candle_plots`
- `--window`：特征窗口大小，默认 `100`

图片会按股票代码保存到：

```text
./candle_plots/sym*/date*.png
```

## 推荐使用流程

1. 准备符合命名规则的 CSV 数据。
2. 执行 `train.py` 生成缓存并训练模型。
3. 执行 `test.py` 查看模型效果和阈值表现。
4. 按需要执行 `analyze_features.py`、`count_max_pnl.py` 或 `draw_candle.py` 做进一步分析。
