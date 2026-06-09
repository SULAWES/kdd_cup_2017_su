# 路线：四模型融合

## 核心思路

四模型融合把几个误差形态不同的预测器做 convex weighted mean。当前实现位于 `src/kddcup2017_task2/ensemble.py`。

当前正式选入 `src/` 的版本沿用上一版四个候选模型，但把融合权重从一组全局权重升级为按目标小时分别学习。上一版全局权重路线可以继续用 `--weight-scope global` 复现，用于对照。

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
3. 按目标小时分别拟合融合权重。
4. 用该权重评估 phase1 验证周 Oct.18-24。

结果：

- calibration MAPE：`0.136831`
- phase1 MAPE：`0.116167`
- 上一版全局权重 phase1 MAPE：`0.118018`

权重：

| 目标小时 | `low_volume_block` | `xgb` | `mlp` | `ratio_lag_7` |
| --- | ---: | ---: | ---: | ---: |
| `08` | `0.857320` | `0.000000` | `0.064316` | `0.078364` |
| `09` | `0.933716` | `0.000000` | `0.000000` | `0.066284` |
| `17` | `0.330386` | `0.301570` | `0.285249` | `0.082795` |
| `18` | `0.309732` | `0.213464` | `0.277083` | `0.199722` |

这一步从上一版 SOTA 到当前 SOTA 的核心变化是：早高峰和晚高峰不再共用同一个融合比例。滚动 train1 折显示，按小时分组的选择比全局权重略优，也比直接从 phase1 结果中挑更激进的 block shrinkage 更容易解释。

## phase2 合法预测路线

命令：

```sh
python run_task2.py predict-ensemble
```

流程：

1. 用 train1 训练，预测 Oct.18-24。
2. 因为 Oct.18-24 在 phase2 已经作为训练标签发布，所以可以合法用它校准融合权重。
3. 用 train1 + train2 训练最终模型，预测 Oct.25-31。

校准 MAPE：`0.111638`。

权重：

| 目标小时 | `low_volume_block` | `xgb` | `mlp` | `ratio_lag_7` |
| --- | ---: | ---: | ---: | ---: |
| `08` | `0.622410` | `0.174285` | `0.039189` | `0.164117` |
| `09` | `0.743852` | `0.009330` | `0.094074` | `0.152744` |
| `17` | `0.625160` | `0.019524` | `0.355316` | `0.000000` |
| `18` | `0.160764` | `0.247434` | `0.116063` | `0.475738` |

输出：

```sh
outputs/submission_task2_volume_ensemble.csv
```

## 泄露边界

`predict-ensemble` 的 `0.111638` 不能当作无泄露 phase1 验证分数，因为它使用了 Oct.18-24 的真实标签调权。它只对 phase2 合法：预测 Oct.25-31 时，Oct.18-24 已经是历史训练数据。

可报告的无泄露 phase1 结果应使用 `validate-ensemble` 的 `0.116167`。上一版全局权重结果可用 `python run_task2.py validate-ensemble --weight-scope global` 复现，为 `0.118018`。
