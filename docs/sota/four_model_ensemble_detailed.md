# 四模型融合方法详细说明

## 定位

当前指标最高的可复现路线是四模型融合，入口在 `run_task2.py`：

```sh
python run_task2.py validate-ensemble
python run_task2.py predict-ensemble
```

实现文件：

- `src/kddcup2017_task2/ensemble.py`
- `src/kddcup2017_task2/pipeline.py`
- `src/kddcup2017_task2/features.py`
- `src/kddcup2017_task2/data.py`

该路线的核心是：先生成 4 个误差形态不同的预测器输出，再用非负权重做加权平均。权重通过历史校准集拟合，不在待评估目标周上调权。

## 指标

无泄露 phase1 验证：

- 命令：`python run_task2.py validate-ensemble`
- 校准集：train1 最后一周
- phase1 目标周：Oct.18-Oct.24
- phase1 MAPE：`0.118018`

phase2 合法预测校准：

- 命令：`python run_task2.py predict-ensemble`
- 校准集：Oct.18-Oct.24 已发布训练标签
- phase2 目标周：Oct.25-Oct.31
- 校准 MAPE：`0.116116`
- 输出：`outputs/submission_task2_volume_ensemble.csv`

注意：`0.116116` 不能作为无泄露 phase1 验证分数。它只表示在 phase2 场景下，使用已经发布的 Oct.18-Oct.24 标签校准融合权重时，历史校准误差达到该水平。

## 代码入口与函数链路

`run_task2.py` 调用 `kddcup2017_task2.pipeline.main()`，命令行子命令再分发到以下函数：

| 命令 | 入口函数 | 目的 |
| --- | --- | --- |
| `validate-ensemble` | `validate_latest_fold_ensemble(args)` | train1 内部校准，验证 Oct.18-Oct.24 |
| `predict-ensemble` | `predict_phase2_ensemble(args)` | 用 train2 合法校准，预测 Oct.25-Oct.31 |

两个入口都会调用同一个候选预测矩阵生成函数：

```text
fit_ensemble_prediction_matrix(
    train_agg,
    known_agg,
    weather,
    train_attr_agg,
    known_attr_agg,
    train_days,
    pred_rows,
    combos,
)
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `train_agg` | 有标签训练期的 20 分钟流量聚合 |
| `known_agg` | 训练聚合加上预测日可见绿窗 |
| `weather` | 天气表，当前基础特征构造里读取但融合路线默认不启用天气特征 |
| `train_attr_agg` | 训练期车辆属性聚合 |
| `known_attr_agg` | 训练期属性加上预测日可见绿窗属性 |
| `train_days` | 用来生成训练样本的日期集合 |
| `pred_rows` | 需要预测的目标行 |
| `combos` | tollgate-direction 组合 |

该函数输出：

```text
matrix: shape = [n_prediction_rows, 4]
predictions: dict[name, vector]
```

矩阵四列顺序固定为：

```text
("low_volume_block", "xgb", "mlp", "ratio_lag_7")
```

## 候选矩阵如何生成

`fit_ensemble_prediction_matrix` 内部执行顺序如下：

1. 调 `train_and_predict(..., group="low_volume_block", transform="log", sample_weight_power=0.3, drop_features=DEFAULT_DROP_FEATURES)` 得到主模型预测。
2. 用 `make_target_rows(train_days, combos)` 生成训练标签行。
3. 从 `train_agg` 取出每行真实红窗流量 `y_train`。
4. 把主流量目标转为 `log1p(y_train)`。
5. 用 `mape_sample_weight(y_train)` 生成近似 MAPE 的样本权重。
6. 构造 `FeatureBuilder(train_agg, weather, include_weather=False)`。
7. 用训练行拟合历史统计量 `builder.fit_stats(train_rows)`。
8. 分别 transform 训练行和预测行。
9. 用 `filter_features(..., DEFAULT_DROP_FEATURES)` 剪掉验证中不稳定的特征。
10. 用 `Vectorizer` 固定特征列顺序，得到 `x_train` 和 `x_pred`。
11. 分别调用 `predict_xgb`、`predict_mlp`、`predict_ratio_lag7`。
12. 按 `ENSEMBLE_MODEL_NAMES` 拼成 4 列预测矩阵。

关键实现点：

- 四个候选模型使用同一批目标行 `pred_rows`，所以输出天然行对齐。
- `Vectorizer.fit_transform` 只在训练特征上拟合列名，预测特征只能按已知列 transform。
- 天气表会被读取，但当前这里显式 `include_weather=False`，避免天气特征在局部验证中引入额外噪声。
- `known_agg` 只应包含预测日合法绿窗，不能包含待预测红窗。

## 四个基础模型

### 1. `low_volume_block`

这是当前最强单模型路线。它先训练 `global` 和 `block` 两个 ExtraTrees 模型，再根据最近 7 天是否进入低流量 regime 来决定某个 combo 是否切到 `block` 预测。

默认配置：

- 模型：`ExtraTreesRegressor`
- 目标：`log1p(volume)`
- 随机种子：`13`
- `max_depth=14`
- `min_samples_leaf=10`
- 样本权重：`sample_weight_power=0.3`
- 特征剪枝：启用 `DEFAULT_DROP_FEATURES`
- 结构切换：当前 phase1 和 phase2 训练数据下均只触发 `1_0`

`low_volume_block` 的意义在于处理局部 regime 变化。普通全局模型会把所有 combo 的经验混在一起；当某个 combo 最近一周明显偏低时，全局模型容易预测偏高。block 模型把早高峰和晚高峰分开训练，能更贴近局部时段形态。

### 2. `xgb`

XGBoost 使用与主模型相同的向量化特征，但模型族不同，用来引入另一类树模型偏差。

主要参数：

- `n_estimators=500`
- `learning_rate=0.025`
- `max_depth=3`
- `min_child_weight=5`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `reg_lambda=1.0`
- `reg_alpha=0.05`
- `objective="reg:squarederror"`
- `random_state=13`

训练目标为 `log1p(volume)`，预测后用 `expm1` 还原并裁剪到非负。

### 3. `mlp`

MLP 是标准化后的浅层神经网络，用于提供非树模型的误差形态。

结构：

- `StandardScaler`
- `MLPRegressor(hidden_layer_sizes=(48, 24))`
- `alpha=0.01`
- `learning_rate_init=0.003`
- `max_iter=1200`
- `early_stopping=True`
- `validation_fraction=0.15`
- `random_state=13`

训练目标同样是 `log1p(volume)`。MLP 没有使用样本权重，主要作为低相关性的补充预测器。

### 4. `ratio_lag_7`

该模型不直接预测流量，而是预测相对 7 天前同一窗口流量的比例。

训练目标：

```text
base = max(lag_7, 1)
target = log((volume + 1) / base)
```

预测还原：

```text
pred = exp(model_output) * max(lag_7, 1) - 1
```

模型配置：

- 模型：`ExtraTreesRegressor`
- `n_estimators=500`
- `max_depth=14`
- `min_samples_leaf=10`
- `random_state=17`
- 样本权重：同 `mape_sample_weight`

它的单模型分数不强，但在融合里能提供周周期相关的补充信号。

## 训练目标与权重对比

| 模型 | 学习目标 | 样本权重 | 主要作用 |
| --- | --- | --- | --- |
| `low_volume_block` | `log1p(volume)` | 有 | 主干预测，处理低流量 combo |
| `xgb` | `log1p(volume)` | 有 | 提供不同树模型偏差 |
| `mlp` | `log1p(volume)` | 无 | 提供非树模型误差形态 |
| `ratio_lag_7` | `log((volume + 1) / max(lag_7, 1))` | 有 | 显式建模周周期比例 |

`mape_sample_weight` 的形式是：

```text
denom = max(y, 1)
weight = (mean(denom) / denom) ** 0.3
weight = weight / mean(weight)
```

这不是直接优化 MAPE，而是在回归器训练阶段提高低流量样本相对权重，缓和 MAPE 对小分母敏感的问题。

## 融合方式

四个模型的预测组成矩阵：

```text
P = [pred_low_volume_block, pred_xgb, pred_mlp, pred_ratio_lag_7]
```

融合预测：

```text
pred = P @ weights
```

权重约束：

- 所有权重非负
- 权重总和等于 `1`
- 目标函数为校准集 MAPE

优化器：

- `scipy.optimize.minimize`
- 方法：`SLSQP`
- 约束：`sum(weights) == 1`
- 边界：每个权重在 `[0, 1]`
- 多个初始点：均匀权重和每个单模型独占权重
- `maxiter=1000`
- `ftol=1e-12`

多初始点的作用是降低 SLSQP 从单个初始点陷入较差局部解的概率。最终选择校准 MAPE 最低的结果。

## `validate-ensemble`

目标：给出无泄露 phase1 验证结果。

数据边界：

```text
train1: Sep.19-Oct.17
test1: Oct.18-Oct.24 绿窗
train2: Oct.18-Oct.24 标签，只用于最后算分
```

流程：

1. `latest_training_fold_split(train_days_all)` 从 train1 中取最后 7 天作为校准集。
2. `filter_days` 和 `filter_attr_days` 得到校准训练部分。
3. `observation_windows_only(train1, calibration_days)` 只取校准日绿窗。
4. `merge_aggregates` 把校准训练历史和校准日绿窗合成 `calibration_known`。
5. 用校准训练部分训练四个模型，预测校准集红窗。
6. 用校准集真实红窗拟合融合权重。
7. 用完整 train1 训练四个模型。
8. 用 train1 + test1 绿窗预测 Oct.18-Oct.24。
9. 用第 6 步权重融合。
10. 最后才用 train2 标签计算 MAPE。

权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.776957` |
| `xgb` | `0.000000` |
| `mlp` | `0.153564` |
| `ratio_lag_7` | `0.069479` |

该流程不使用 Oct.18-Oct.24 标签拟合权重，因此可以作为无泄露验证结果。

## `predict-ensemble`

目标：生成 phase2 融合提交。

数据边界：

```text
train1: Sep.19-Oct.17
test1: Oct.18-Oct.24 绿窗
train2: Oct.18-Oct.24 标签，phase2 已发布
test2: Oct.25-Oct.31 绿窗
```

流程：

1. 用 train1 训练四个模型。
2. 用 train1 + test1 绿窗预测 Oct.18-Oct.24。
3. 使用已发布的 train2 标签拟合融合权重。
4. 合并 train1 + train2，得到最终训练标签。
5. 合并最终训练标签和 test2 绿窗，得到 phase2 可见输入。
6. 读取 `submission_sample_volume.csv` 的样例结构。
7. 用 `make_target_rows_like_sample` 把样例日期平移到 test2 第一预测日。
8. 训练最终四个模型并预测 Oct.25-Oct.31。
9. 用第 3 步权重融合并写出提交。

权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.512492` |
| `xgb` | `0.115189` |
| `mlp` | `0.153896` |
| `ratio_lag_7` | `0.218422` |

该流程对 phase2 是合法的，因为 Oct.18-Oct.24 在 phase2 中已经是历史训练标签。

## 为什么两个模式权重不同

`validate-ensemble` 的权重来自 train1 内部最后一周，目标是模拟“未来一周完全未知”的 phase1 验证。这个校准集更早，并且仍受 Sep.19-Oct.17 内部节假日和分布变化影响。

`predict-ensemble` 的权重来自 Oct.18-Oct.24，它更接近 Oct.25-Oct.31 的时间位置。对 phase2 来说，这一周标签已经发布，因此调权合法。

这解释了两个现象：

- `xgb` 在无泄露 phase1 权重中为 0，但在 phase2 合法校准中为非零。
- `ratio_lag_7` 在 phase2 权重中更高，说明 Oct.18-Oct.24 到 Oct.25-Oct.31 的周周期校准信号更强。

## 为什么弱单模型仍能进入融合

融合不是单模型排行榜。一个模型是否有价值，取决于它的残差是否能抵消主模型残差。

例如：

- 主模型在某些低流量窗口偏高，`ratio_lag_7` 可能更贴近上一周比例。
- ExtraTrees 对某些离散特征组合稳定，XGBoost 可能在另一些连续统计特征上更平滑。
- MLP 的预测可能整体不够准，但如果它在少数高 MAPE 样本上的方向正确，融合器会给它小权重。

非负且和为 1 的约束可以防止融合器用过大的负权重或杠杆权重制造不稳定预测。

## 输出和复现检查

运行 `validate-ensemble` 后，应关注：

- `calibration=latest_training_fold`
- `leakage_check=uses only labels before validation period`
- `validation_rows=420`
- `validation_mape=0.118018`

运行 `predict-ensemble` 后，应关注：

- `calibration=train1_to_train2`
- `leakage_check=legal_for_phase2_only; do not report this as a no-leak phase1 validation score`
- `prediction_rows=420`
- `submission=outputs/submission_task2_volume_ensemble.csv`

如果本地结果有明显差异，优先检查：

1. 依赖版本是否缺失，尤其是 `xgboost`。
2. 是否正确设置 `.codex_deps` 或安装了同等依赖。
3. 数据目录是否仍为默认 `dataset/`。
4. `submission_sample_volume.csv` 是否存在且未被修改。
5. 是否误把 `predict-ensemble` 的校准分数当成验证分数。

## 依赖

除基础运行依赖外，四模型融合需要：

- `scikit-learn`
- `scipy`
- `xgboost`

当前本地实验把依赖安装在 `.codex_deps/`，该目录已在 `.gitignore` 中忽略。

## 风险与解释

1. 融合权重对校准集分布敏感。国庆异常周会使权重明显偏移，因此 `validate-ensemble` 只采用最近训练周校准，而不是混合所有旧滚动折。
2. `xgb` 和 `mlp` 单模型分数较弱，但与主模型误差相关性不完全相同，因此在特定校准下仍能贡献权重。
3. `predict-ensemble` 的校准分数更低，但它使用了 train2 标签；这个数只说明 phase2 合法校准效果，不能用于宣传 phase1 无泄露验证成绩。
4. 当前天气表读取但不启用天气特征。之前的局部验证没有证明天气特征能稳定提升，因此 SOTA 路线保留了更保守的特征集。
5. 当前没有使用 trajectory 表。若后续引入，必须按预测时间切分，只允许使用红窗之前可见的轨迹信息。
