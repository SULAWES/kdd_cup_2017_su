# 路线：四模型融合

## 核心思路

四模型融合把几个误差形态不同的预测器做 convex weighted mean。当前实现位于 `src/kddcup2017_task2/ensemble.py`。

更详细的实现说明见：

- [四模型融合入门说明](sota/four_model_ensemble_getting_started.md)
- [四模型融合方法详细说明](sota/four_model_ensemble_detailed.md)
- [四模型融合的数据处理与数据使用说明](sota/four_model_ensemble_data_usage.md)

候选模型：

1. `low_volume_block`
2. `xgb`
3. `mlp`
4. `ratio_lag_7`

其中：

- `low_volume_block` 是当前最佳单模型结构路线。
- `xgb` 提供另一类树模型偏差。
- `mlp` 是标准化后的浅层神经网络。
- `ratio_lag_7` 不直接预测流量，而预测相对 `lag_7` 的比例，再乘回 `lag_7`。

## 无泄露验证路线

命令：

```sh
python run_task2.py validate-ensemble
```

流程：

1. 用训练期最后一周做校准折。
2. 在校准折上拟合四模型融合权重。
3. 用该权重评估 phase1 验证周 Oct.18-24。

结果：

- calibration MAPE：`0.141467`
- phase1 MAPE：`0.118018`

权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.776957` |
| `xgb` | `0.000000` |
| `mlp` | `0.153564` |
| `ratio_lag_7` | `0.069479` |

## phase2 合法预测路线

命令：

```sh
python run_task2.py predict-ensemble
```

流程：

1. 用 train1 训练，预测 Oct.18-24。
2. 因为 Oct.18-24 在 phase2 已经作为训练标签发布，所以可以合法用它校准融合权重。
3. 用 train1 + train2 训练最终模型，预测 Oct.25-31。

校准 MAPE：`0.116116`。

权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.512492` |
| `xgb` | `0.115189` |
| `mlp` | `0.153896` |
| `ratio_lag_7` | `0.218422` |

输出：

```sh
outputs/submission_task2_volume_ensemble.csv
```

## 泄露边界

`predict-ensemble` 的 `0.116116` 不能当作无泄露 phase1 验证分数，因为它使用了 Oct.18-24 的真实标签调权。它只对 phase2 合法：预测 Oct.25-31 时，Oct.18-24 已经是历史训练数据。

可报告的无泄露 phase1 结果应使用 `validate-ensemble` 的 `0.118018`。
