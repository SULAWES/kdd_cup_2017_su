# 四模型融合入门说明

这份文档面向第一次接触本项目的人，目标是说明当前最好方案在做什么、为什么这样做、如何合法复现结果，以及运行输出应该如何解读。

如果只想快速得到结论：

- 无泄露 phase1 验证命令：`python run_task2.py validate-ensemble`
- phase2 提交生成命令：`python run_task2.py predict-ensemble`
- 无泄露 phase1 MAPE：`0.118018`
- phase2 合法校准 MAPE：`0.116116`
- phase2 提交文件：`outputs/submission_task2_volume_ensemble.csv`

`0.116116` 是用已经发布的 Oct.18-Oct.24 标签调融合权重后的历史校准误差，只能说明 phase2 场景下的合法校准效果，不能当作无泄露 phase1 验证分数。

## 任务是什么

KDD Cup 2017 Task 2 的目标是预测收费站在未来高峰期的 20 分钟平均车流量。

需要预测的对象有 5 个收费站/方向组合：

| combo | 含义 |
| --- | --- |
| `1_0` | tollgate 1, direction 0 |
| `1_1` | tollgate 1, direction 1 |
| `2_0` | tollgate 2, direction 0 |
| `3_0` | tollgate 3, direction 0 |
| `3_1` | tollgate 3, direction 1 |

每天需要预测 12 个时间窗口：

- 早高峰：`08:00-10:00`，共 6 个 20 分钟窗口
- 晚高峰：`17:00-19:00`，共 6 个 20 分钟窗口

一周提交行数：

```text
7 天 * 12 个窗口 * 5 个组合 = 420 行
```

评价指标是 MAPE：

```text
mean(abs(actual - pred) / actual)
```

它会放大低流量样本的相对误差。例如实际流量为 10 时预测偏 5，会贡献 50% 相对误差；实际流量为 100 时同样偏 5，只贡献 5% 相对误差。因此模型不能只追求普通均方误差低，也要照顾低流量 combo 和低流量窗口。

## 数据怎么分

本项目按比赛数据交换后的结构理解数据：

| 数据 | 日期 | 在本项目中的角色 |
| --- | --- | --- |
| train1 | Sep.19-Oct.17 | 初始训练标签 |
| test1 | Oct.18-Oct.24 | phase1 绿窗输入 |
| train2 | Oct.18-Oct.24 | phase2 新增训练标签 |
| test2 | Oct.25-Oct.31 | phase2 绿窗输入 |

这里的“绿窗输入”指题目明确给出的当天更早时间段流量：

- 预测早高峰时，可使用 `06:00-08:00`
- 预测晚高峰时，可使用 `15:00-17:00`

这些绿窗数据是合法输入。红窗真实流量才是要预测的目标，不能在预测时使用。

可以把一天里的可用信息理解为：

```text
早高峰预测:
06:00-08:00  已知绿窗
08:00-10:00  待预测红窗

晚高峰预测:
15:00-17:00  已知绿窗
17:00-19:00  待预测红窗
```

## 当前最好方案一眼看懂

当前最好方案是四模型融合。整体链路如下：

```text
CSV 原始流量
  -> 20 分钟窗口聚合
  -> 构造每个 combo / 日期 / 目标窗口的特征行
  -> 训练 4 个候选预测器
  -> 在历史校准集上学习非负融合权重
  -> 对目标周生成 420 行预测
```

四个候选预测器分别是：

```text
low_volume_block
xgb
mlp
ratio_lag_7
```

融合形式是一个非负加权平均：

```text
final_pred =
  w1 * low_volume_block
  + w2 * xgb
  + w3 * mlp
  + w4 * ratio_lag_7
```

权重约束：

- 每个权重不小于 0
- 所有权重加起来等于 1
- 权重由校准集上的 MAPE 最小化得到

这样做的目的不是让 4 个模型都很强，而是让它们的错误不完全相同。主模型负责大部分精度，其他模型在特定 combo、特定窗口或周周期变化上补偏差。

## 应该先跑哪个命令

建议顺序：

1. 先跑 `python run_task2.py validate-ensemble`
2. 确认无泄露验证结果和文档一致。
3. 再跑 `python run_task2.py predict-ensemble`
4. 用生成的 `outputs/submission_task2_volume_ensemble.csv` 做 phase2 提交。

两个命令的职责不同：

| 命令 | 用途 | 是否可作为无泄露验证 |
| --- | --- | --- |
| `validate-ensemble` | 用 train1 内部最后一周校准，然后验证 Oct.18-Oct.24 | 是 |
| `predict-ensemble` | 用 Oct.18-Oct.24 已发布标签校准，然后预测 Oct.25-Oct.31 | 否，但对 phase2 提交合法 |

不要把 `predict-ensemble` 的校准分数拿来宣传为无泄露 phase1 分数。

## 四个模型分别负责什么

### `low_volume_block`

这是主力模型。

它基于 ExtraTrees，并加入一个结构判断：如果某个收费站/方向组合最近 7 天明显低于整体均值和自身历史均值，就改用早晚块模型预测该组合。

当前数据下，这个规则会选择 `1_0`。

直觉上，`1_0` 在训练末期出现低位 regime，用普通全局模型会被其他组合的正常水平拉高；单独切换到 block 模型能降低这部分误差。

### `xgb`

这是 XGBoost 模型。它使用同一批基础特征，但模型族不同，误差形态与 ExtraTrees 不完全一样。

单独看它分数不如主模型，但融合时可能补充一部分主模型的偏差。它在 phase2 合法校准中获得了非零权重，说明它对 Oct.18-Oct.24 这一周的预测误差和主模型存在可利用差异。

### `mlp`

这是一个标准化后的浅层神经网络。

结构是：

```text
StandardScaler -> MLPRegressor(48, 24)
```

它同样不是最强单模型，但提供了非树模型的预测形态。融合器会自己决定它是否值得保留权重。

### `ratio_lag_7`

这个模型预测的不是流量本身，而是相对 7 天前同一窗口的比例：

```text
target = log((volume + 1) / max(lag_7, 1))
```

预测时再乘回 `lag_7`：

```text
pred = exp(model_output) * max(lag_7, 1) - 1
```

它的作用是显式引入周周期。Task 2 的目标窗口以一周为提交单位，`lag_7` 往往比 `lag_1` 更贴近相同 weekday 和相同高峰块的通行模式。

## `validate-ensemble` 具体做什么

这是无泄露验证命令。

它的流程是：

1. 从 train1 中拿最后 7 天作为校准集。
2. 用更早的 train1 数据训练四个模型。
3. 预测校准集红窗，拟合融合权重。
4. 用完整 train1 训练四个模型。
5. 用 test1 绿窗预测 Oct.18-Oct.24。
6. 用第 3 步得到的权重融合。
7. 最后才用 train2 标签计算 MAPE。

关键点：train2 标签没有参与权重拟合。

当前结果：

```text
phase1 MAPE = 0.118018
```

当前权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.776957` |
| `xgb` | `0.000000` |
| `mlp` | `0.153564` |
| `ratio_lag_7` | `0.069479` |

这是可报告的无泄露 phase1 验证结果。

## `predict-ensemble` 具体做什么

这是 phase2 提交命令。

它的流程是：

1. 用 train1 训练四个模型。
2. 用 test1 绿窗预测 Oct.18-Oct.24。
3. 用 train2 标签拟合融合权重。
4. 用 train1 + train2 训练最终四个模型。
5. 用 test2 绿窗预测 Oct.25-Oct.31。
6. 写出提交文件。

当前校准结果：

```text
calibration MAPE = 0.116116
```

当前权重：

| 模型 | 权重 |
| --- | ---: |
| `low_volume_block` | `0.512492` |
| `xgb` | `0.115189` |
| `mlp` | `0.153896` |
| `ratio_lag_7` | `0.218422` |

这个数不能当作无泄露 phase1 验证分数，因为它用到了 Oct.18-Oct.24 标签调权。但在 phase2 预测 Oct.25-Oct.31 时，Oct.18-Oct.24 已经是公开训练数据，所以这种校准是合法的。

输出文件：

```text
outputs/submission_task2_volume_ensemble.csv
```

## 如何从零复现

在仓库根目录运行：

```sh
python run_task2.py validate-ensemble
```

如果本机默认 Python 没有依赖，需要准备：

- `numpy`
- `scikit-learn`
- `scipy`
- `xgboost`

当前 Codex 环境中实验用依赖放在 `.codex_deps/`，运行时可设置：

```powershell
$env:PYTHONPATH='D:\Dev\kdd_cup_2017_su\.codex_deps'
python run_task2.py validate-ensemble
```

生成 phase2 融合提交：

```sh
python run_task2.py predict-ensemble
```

如果需要同时设置依赖路径：

```powershell
$env:PYTHONPATH='D:\Dev\kdd_cup_2017_su\.codex_deps'
python run_task2.py predict-ensemble
```

## 如何读终端输出

`validate-ensemble` 里比较重要的输出字段：

| 字段 | 含义 |
| --- | --- |
| `calibration=latest_training_fold` | 权重来自 train1 内部最后一周校准 |
| `leakage_check=uses only labels before validation period` | 表示权重没有用 Oct.18-Oct.24 标签 |
| `calibration_rows=420` | 校准集行数，一周 420 行 |
| `calibration_mape` | train1 内部校准误差 |
| `weight_*` | 四个候选模型的融合权重 |
| `single_*_mape` | 每个候选模型在 phase1 验证周上的单模型 MAPE |
| `validation_rows=420` | phase1 验证行数 |
| `validation_mape` | 最终无泄露 phase1 MAPE |
| `actual_mean` / `pred_mean` | 验证标签均值和预测均值，用于发现整体偏高或偏低 |

`predict-ensemble` 里比较重要的输出字段：

| 字段 | 含义 |
| --- | --- |
| `calibration=train1_to_train2` | 用 train1 预测 train2，再用 train2 标签调权 |
| `leakage_check=legal_for_phase2_only` | 表示该校准只对 phase2 提交合法 |
| `calibration_rows=420` | Oct.18-Oct.24 校准行数 |
| `calibration_mape` | phase2 合法历史校准误差 |
| `weight_*` | 用于 phase2 提交的融合权重 |
| `prediction_rows=420` | phase2 提交行数 |
| `pred_mean` | phase2 提交预测均值 |
| `submission` | 写出的提交文件路径 |

## 为什么要单独强调数据使用

这个项目里最容易犯错的地方不是模型代码，而是验证边界。

可以使用：

- 训练期历史红窗标签
- 测试日题目给出的绿窗输入
- phase2 已发布的 Oct.18-Oct.24 标签，用于预测 Oct.25-Oct.31

不可以使用：

- 待预测红窗真实标签
- 用 Oct.18-Oct.24 标签调权后，再把结果说成无泄露 phase1 验证
- 预测某日早高峰时使用 `08:00` 之后真实流量
- 预测某日晚高峰时使用 `17:00` 之后真实流量

因此，本项目把两个命令分开：

- `validate-ensemble` 用来做无泄露验证
- `predict-ensemble` 用来做 phase2 合法校准和提交

更详细的数据处理说明见 `docs/sota/four_model_ensemble_data_usage.md`。

## 常见误区

### 误区 1：四个模型都必须单独很强

不需要。融合关注的是组合后的误差，只要某个弱模型的误差方向和主模型不同，它就可能在非负权重约束下有价值。

### 误区 2：`predict-ensemble` 的 0.116116 就是最终公开榜分数

不是。它是 Oct.18-Oct.24 上的校准误差。真正 phase2 目标是 Oct.25-Oct.31，这一周没有本地标签，只能生成提交文件。

### 误区 3：test 文件都不能用

不对。题目明确给出了测试日绿窗输入，预测红窗时可以用这些绿窗。不能用的是目标红窗真实流量。

### 误区 4：提交文件随便按日期和 combo 排序即可

不建议。当前实现读取 `submission_sample_volume.csv` 的形状并平移日期，以保持样例提交的行结构，避免格式或顺序错配。

## 继续改进可以从哪里下手

优先方向：

1. 做更稳健的权重校准，不只依赖最后一周。
2. 给不同 combo 学不同融合权重，但必须用训练折估计。
3. 对 `ratio_lag_7` 加入更稳定的基线，例如 combo-slot 历史中位数。
4. 分析 `1_0` 低位 regime 是否有更明确的触发特征。
5. 如果引入 trajectory 表，必须只使用预测时间之前可见的数据，并先做严格验证。
