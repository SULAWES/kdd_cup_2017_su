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

该路线的核心是：先生成 4 个误差形态不同的预测器输出，再用非负权重做加权平均。权重通过历史校准集拟合，不在待评估目标周上调权。

## 指标

无泄露 phase1 验证：

- 命令：`python run_task2.py validate-ensemble`
- 校准集：训练期最后一周
- phase1 MAPE：`0.118018`

phase2 合法预测校准：

- 命令：`python run_task2.py predict-ensemble`
- 校准集：Oct.18-Oct.24 已发布训练标签
- 校准 MAPE：`0.116116`
- 输出：`outputs/submission_task2_volume_ensemble.csv`

注意：`0.116116` 不能作为无泄露 phase1 验证分数。它只表示在 phase2 场景下，使用已经发布的 Oct.18-Oct.24 标签校准融合权重时，历史校准误差达到该水平。

## 四个基础模型

### 1. `low_volume_block`

这是当前最强单模型路线。它先训练 `global` 和 `block` 两个 ExtraTrees 模型，再根据最近 7 天是否进入低流量 regime 来决定某个 combo 是否切到 `block` 预测。

默认配置：

- 模型：`ExtraTreesRegressor`
- 目标：`log1p(volume)`
- 样本权重：`sample_weight_power=0.3`
- 特征剪枝：启用 `DEFAULT_DROP_FEATURES`
- 结构切换：当前 phase1 和 phase2 训练数据下均只触发 `1_0`

该模型提供融合中的主干预测。

### 2. `xgb`

XGBoost 使用与主模型相同的向量化特征，但模型族不同，用来引入另一类树模型偏差。

主要参数：

- `n_estimators=500`
- `learning_rate=0.025`
- `max_depth=3`
- `min_child_weight=5`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `objective="reg:squarederror"`

训练目标为 `log1p(volume)`，预测后用 `expm1` 还原。

### 3. `mlp`

MLP 是标准化后的浅层神经网络，用于提供非树模型的误差形态。

结构：

- `StandardScaler`
- `MLPRegressor(hidden_layer_sizes=(48, 24))`
- `alpha=0.01`
- `learning_rate_init=0.003`
- `early_stopping=True`

训练目标同样是 `log1p(volume)`。

### 4. `ratio_lag_7`

该模型不直接预测流量，而是预测相对 7 天前同一窗口流量的比例。

训练目标：

```text
log((volume + 1) / max(lag_7, 1))
```

预测还原：

```text
pred = exp(model_output) * max(lag_7, 1) - 1
```

模型使用 `ExtraTreesRegressor`，随机种子为 `17`。它的单模型分数不强，但在融合里能提供周周期相关的补充信号。

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
- 多个初始点：均匀权重和每个单模型独占权重

## 两种运行模式

### `validate-ensemble`

目标：给出无泄露 phase1 验证结果。

流程：

1. 从 train1 中取最后 7 天作为校准集。
2. 用更早训练日训练四个模型。
3. 用校准集当天绿窗预测校准集红窗。
4. 用校准集真实红窗拟合融合权重。
5. 用完整 train1 训练四个模型。
6. 用 phase1 test 绿窗预测 Oct.18-Oct.24。
7. 用第 4 步权重融合，并与 train2 标签计算 MAPE。

权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.776957` |
| `xgb` | `0.000000` |
| `mlp` | `0.153564` |
| `ratio_lag_7` | `0.069479` |

该流程不使用 Oct.18-Oct.24 标签拟合权重，因此可以作为无泄露验证结果。

### `predict-ensemble`

目标：生成 phase2 融合提交。

流程：

1. 用 train1 训练四个模型。
2. 用 phase1 test 绿窗预测 Oct.18-Oct.24。
3. 使用已发布的 train2 标签拟合融合权重。
4. 用 train1 + train2 训练最终四个模型。
5. 用 phase2 test 绿窗预测 Oct.25-Oct.31。
6. 用第 3 步权重融合并写出提交。

权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.512492` |
| `xgb` | `0.115189` |
| `mlp` | `0.153896` |
| `ratio_lag_7` | `0.218422` |

该流程对 phase2 是合法的，因为 Oct.18-Oct.24 在 phase2 中已经是历史训练标签。

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
