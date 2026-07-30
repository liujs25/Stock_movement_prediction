# Stock Movement Prediction

> 基于高频订单簿快照的多时间尺度价格移动方向预测：从数据质量检查、特征工程和时序验证，到 XGBoost 训练、阈值推理与平台提交的完整工程流程。

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-FF6600)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-6A5ACD)
![Status](https://img.shields.io/badge/Status-Final%20Submission-2E8B57)

## 项目概览

本项目使用 tick 级限价订单簿快照，预测未来 `5 / 10 / 20 / 40 / 60` 个 tick 的价格移动方向。每个预测 horizon 都是一个三分类任务：

- `0`：下跌
- `1`：价格基本不变
- `2`：上涨

项目围绕金融时序建模中最关键的三类问题展开：

1. **数据可信度**：检查文件完整性、订单簿异常、midprice 一致性和标签分布。
2. **防止时间泄露**：所有特征只使用当前及历史 tick，并采用基于日期的 out-of-time 验证。
3. **模型到提交的一致性**：保证训练与推理阶段的特征名称、顺序和数值口径一致，并提供可直接提交的平台包。

最终主线版本采用 **Date63 holdout + XGBoost + 分 horizon 阈值 + turbo 批量推理**。五个任务的平台 `total PnL` 合计为 **40.60**，其中 `label_20`、`label_40` 和 `label_60` 获得正 `model score`。

## 核心结果

| Horizon | Precision | Recall | F0.5 | Total PnL | Single PnL | Model Score |
|---|---:|---:|---:|---:|---:|---:|
| `label_5` | 0.8014 | 0.0518 | 0.2058 | -0.42 | -0.000054 | -0.000425 |
| `label_10` | 0.7625 | 0.0660 | 0.2450 | 2.10 | 0.000147 | -0.000156 |
| `label_20` | 0.6233 | 0.0694 | 0.2400 | 11.72 | 0.000761 | 0.000312 |
| `label_40` | 0.5919 | 0.0412 | 0.1612 | 14.61 | 0.001104 | **0.000798** |
| `label_60` | 0.5819 | 0.0289 | 0.1204 | 12.59 | **0.001179** | 0.000731 |

关键观察：

- `label_40` 的 `model score` 最高。
- `label_60` 的单笔 PnL 最高。
- 短 horizon 虽然 precision 较高，但交易空间不足，score 仍低于平台基准。
- 过高的置信阈值会让信号过度稀疏；金融预测不能只追求 precision，还要联合考虑 recall、覆盖率、F0.5 与 PnL。

> 指标来自最终平台版本 `date63_holdout_turbo_fixed`。完整实验口径与三版结果对比见[最终答辩报告](reports/基于股票快照的价格移动方向预测_最终答辩报告.pdf)。

## 方法与工程流程

```mermaid
flowchart LR
    A["订单簿快照 CSV"] --> B["数据质量检查与 EDA"]
    B --> C["182 个基础因子"]
    C --> D["5 帧拼接与多窗口摘要"]
    D --> E["922 维最终特征"]
    E --> F["5 个 XGBoost 三分类模型"]
    F --> G["分 horizon 置信阈值"]
    G --> H["Turbo 推理与平台提交"]
```

### 数据与验证

- 成功读取 `1,521` 个行情文件，共约 `3.04M` 条原始记录。
- 原始数据包含 10 只股票、79 个交易日与 2 个 session 的组合；存在 59 个缺失文件组合。
- 数据按股票、日期和 session 内的真实时间顺序处理，rolling/lag 特征不会跨边界计算。
- 主线验证使用 `--split-mode date --val-start-date 63`，避免随机划分造成的相邻时段泄露。

### 特征工程

特征体系从 `182` 个基础因子出发，通过最近 5 帧完整特征拼接和 `5 / 10 / 20 / 40 / 80 / 100` tick 历史摘要，形成 `922` 维最终输入。主要包括：

- 原始价格、五档买卖价与挂单量
- spread、depth、imbalance、micro price
- lag、rolling statistics、ROC、volatility、price slope
- OFI、VPIN 类指标、order-book pressure、成交强度
- session 内时间位置和订单簿异常标记

标签、未来价格以及 `date / time / source_file` 等追踪字段不会进入模型特征。

### 模型与推理

- 每个 horizon 单独训练一个 XGBoost 三分类模型。
- 目标函数为 `multi:softprob`，输出下跌、不变、上涨三类概率。
- 推理阶段使用分 horizon 阈值筛选方向信号，低置信度样本回退为“不变”。
- 最终包通过训练-推理 feature parity、256-window 一致性测试和平台 smoke test。

## 仓库结构

```text
.
├── code/
│   ├── Submission_XGBoost/       # 当前主线：训练、构建提交包、推理与 smoke test
│   ├── Reproduction_XGBoost/     # 对早期报告结果的复现实验
│   ├── EDA/                      # EDA 源码、统计结果与图表
│   ├── Previous_version_reference/ # 早期实现，仅作参考
│   ├── submission_template/      # 平台提交接口模板
│   └── tools/                    # 辅助评估与文档工具
├── deliverables/                 # 最终可提交 ZIP
├── docs/                         # 项目报告、因子定义与答辩材料
├── reference_specs/              # 任务规则、数据说明与历史报告
├── reports/                      # EDA、特征工程和最终答辩报告
├── selected_artifacts/           # 精选模型指标、阈值与特征规格
├── src/                          # 早期 EDA/表格特征流程
└── HANDOFF_README.md             # 交接范围、校验值与详细运行说明
```

## 快速开始

### 1. 创建环境

主线代码要求 Python 3.10：

```bash
cd code/Submission_XGBoost
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU 训练环境可改用：

```bash
pip install -r requirements_gpu.txt
```

### 2. 准备数据

由于体积与使用限制，仓库**不包含原始 CSV 数据**。请将数据放在本地目录，并通过 `--data-dir` 显式传入：

```text
/path/to/FBDQA2021A_MMP_Challenge/data/
```

### 3. 运行小规模训练检查

先使用单一标签和少量文件验证环境与数据格式：

```bash
python src/train.py \
  --data-dir /path/to/FBDQA2021A_MMP_Challenge/data \
  --labels label_60 \
  --max-files 80 \
  --num-boost-round 20 \
  --early-stopping-rounds 5
```

### 4. 运行主线日期切分训练

```bash
python src/train.py \
  --data-dir /path/to/FBDQA2021A_MMP_Challenge/data \
  --split-mode date \
  --val-start-date 63 \
  --device cpu
```

训练产物默认写入 `artifacts/`：

- `models/model_label_*.json`
- `feature_spec.json`
- `thresholds.json`
- `metrics/*_thresholds.json`
- `train_summary.json`

### 5. 构建并检查提交包

```bash
python src/build_submission.py
python src/smoke_test_submission.py \
  --data-dir /path/to/FBDQA2021A_MMP_Challenge/data
```

构建脚本会生成扁平提交目录 `artifacts/submission_package/` 和 ZIP 文件 `artifacts/submission_xgboost.zip`。

## 最终提交包

仓库已经包含最终平台提交文件：

```text
deliverables/SUBMIT_THIS_date63_holdout_turbo_fixed.zip
SHA256: 2b47ae87fffdd04ecd926a66327dcc96e540de6717de8676962ea25e887cf65e
```

该压缩包包含平台需要的 `Predictor.py`、特征构造逻辑、配置、五个模型文件、特征规格和阈值文件。它已完成 256-window 零差异一致性检查。

## 文档索引

- [最终答辩报告（PDF）](reports/基于股票快照的价格移动方向预测_最终答辩报告.pdf)
- [项目技术报告](docs/Project_Report.md)
- [特征与因子定义](docs/Feature_Factor_Definitions.md)
- [EDA 分析报告](reports/基于股票快照的价格移动方向预测_EDA_分析报告.pdf)
- [特征工程报告](reports/基于股票快照的价格移动方向预测_特征工程报告.pdf)
- [主线训练与提交说明](code/Submission_XGBoost/README.md)
- [项目交接说明](HANDOFF_README.md)

## 复现说明与限制

- 完整训练依赖未随仓库发布的原始行情 CSV；仅克隆仓库无法直接复现全量训练。
- 不同硬件、XGBoost 版本和线程配置可能带来轻微数值差异，建议使用仓库锁定的依赖版本。
- 平台结果来自特定时间切分与评测规则，不代表真实交易环境中的未来收益。
- 当前流程尚未完整建模手续费、滑点、市场冲击和市场状态漂移。
- 本项目用于课程研究与工程复现，不构成投资建议。

## 后续方向

- 从方向概率进一步转向 expected PnL / positive-EV 信号筛选。
- 对 `softprob` 输出进行概率校准或学习排序。
- 强化 OFI、报价更新速度、成交冲击等短周期微观结构特征。
- 针对长 horizon 优化趋势与跨窗口一致性，提高高收益信号召回率。
- 将 feature parity、batch benchmark 和 smoke test 纳入自动化验证。
