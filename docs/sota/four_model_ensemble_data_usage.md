# 四模型融合的数据处理与数据使用说明

这份文档专门说明当前 SOTA 四模型融合路线如何处理数据、每个运行模式分别使用哪些数据，以及哪些做法会造成泄露。

核心原则：

- 训练标签只能来自当前场景已经发布的训练数据。
- 预测目标红窗时，只能使用题目允许的同日绿窗和更早历史。
- `validate-ensemble` 要模拟 phase1 无泄露验证，不能用 train2 标签调权。
- `predict-ensemble` 面向 phase2，允许使用已经发布的 train2 标签调权。
- 当前默认按目标小时学习融合权重；目标小时来自提交行本身，不是标签，也不包含未来真实流量。

## 数据文件

四模型融合使用的 Task 2 数据文件如下：

| 名称 | 文件 | 用途 |
| --- | --- | --- |
| train1 volume | `dataset/dataSets/training/volume(table 6)_training.csv` | Sep.19-Oct.17 训练标签和历史特征 |
| train2 volume | `dataset/dataSet_phase2/volume(table 6)_training2.csv` | Oct.18-Oct.24 phase2 新增训练标签 |
| test1 volume | `dataset/dataSets/testing_phase1/volume(table 6)_test1.csv` | Oct.18-Oct.24 绿窗输入 |
| test2 volume | `dataset/dataSet_phase2/volume(table 6)_test2.csv` | Oct.25-Oct.31 绿窗输入 |
| sample volume | `dataset/submission_sample_volume.csv` | 输出行顺序和提交格式 |
| weather | `weather (table 7)_*.csv` | 当前融合路线读取但默认不作为模型特征使用 |

这些路径由 `project_paths(data_dir)` 统一生成。默认 `data_dir` 是 `dataset/`。

当前四模型融合没有使用 trajectory 表。之前试验过 trajectory 绿窗统计，但没有优于当前默认路线。若后续重新引入 trajectory，必须先定义每条轨迹在预测时刻是否已经可见，不能直接把目标红窗之后的轨迹统计拼进特征。

## 数据血缘总览

| 运行模式 | 权重校准标签 | 权重粒度 | 基础模型最终训练标签 | 目标预测输入 | 最终本地评分标签 |
| --- | --- | --- | --- | --- | --- |
| `validate-ensemble` | train1 内部最后一周 | 目标小时 | 完整 train1 | test1 绿窗 | train2，仅最后算分 |
| `predict-ensemble` | train2 | 目标小时 | train1 + train2 | test2 绿窗 | 无本地标签 |

这张表是判断是否泄露的最短路径：

- 如果 train2 标签参与 `validate-ensemble` 的权重、模型或特征统计，就是泄露。
- 如果 test2 红窗标签参与任何 phase2 预测，就是泄露。
- 如果 test1/test2 绿窗参与对应目标周预测，是题目允许的输入，不是泄露。

## 20 分钟聚合

原始 volume 记录是一车一行。处理步骤：

1. 读取每条记录的时间字段：
   - `time`
   - 或兼容字段 `date_time`
2. 使用 `floor_20min` 向下取整到 20 分钟窗口。
3. 按 `(window_start, tollgate_id, direction)` 计数。

聚合结果类型：

```text
WindowKey = (datetime, tollgate_id, direction)
value = volume_count
```

例如 `2016-10-18 06:37:12` 会进入 `2016-10-18 06:20:00` 窗口。

窗口按半开区间理解：

```text
[06:20, 06:40)
```

因此恰好 `06:40:00` 的记录进入下一个窗口 `06:40`，不会重复计数。

## 目标窗口

每个预测日有 12 个红窗：

- 早高峰：`08:00`, `08:20`, `08:40`, `09:00`, `09:20`, `09:40`
- 晚高峰：`17:00`, `17:20`, `17:40`, `18:00`, `18:20`, `18:40`

预测对象为 5 个 tollgate-direction 组合：

- `1_0`
- `1_1`
- `2_0`
- `3_0`
- `3_1`

因此 7 天提交共：

```text
7 days * 12 windows * 5 combos = 420 rows
```

目标行在代码中是 `TargetRow`，可以理解为：

```text
TargetRow(day, slot_time, tollgate_id, direction)
```

## 绿窗输入

题目允许在预测红窗时使用同日更早的绿窗流量：

- 预测早高峰红窗时使用 `06:00-08:00`
- 预测晚高峰红窗时使用 `15:00-17:00`

代码中对应：

```text
OBS_TIMES["morning"] = 06:00, 06:20, 06:40, 07:00, 07:20, 07:40
OBS_TIMES["evening"] = 15:00, 15:20, 15:40, 16:00, 16:20, 16:40
```

这些绿窗输入在测试文件中给出，是合法输入，不属于泄露。

一个早高峰目标行的输入边界示例：

```text
预测 2016-10-25 08:40 的 combo 1_0:
可用: 2016-10-25 06:00-08:00 的 test2 绿窗
可用: Oct.24 及之前已经发布的训练标签
不可用: 2016-10-25 08:00-10:00 的红窗真实流量
不可用: 2016-10-25 10:00 之后的任何未来信息
```

## `known_agg` 是什么

`known_agg` 是特征构造时看到的“已知流量世界”。它不是全部真实流量，而是：

```text
已发布训练聚合 + 当前预测日合法绿窗聚合
```

在不同模式中含义不同：

| 场景 | `train_agg` | `known_agg` |
| --- | --- | --- |
| train1 内部校准 | 校准训练日前的 train1 | 校准训练日前的 train1 + 校准日绿窗 |
| phase1 验证 | 完整 train1 | 完整 train1 + test1 绿窗 |
| phase2 校准 | 完整 train1 | 完整 train1 + test1 绿窗 |
| phase2 预测 | train1 + train2 | train1 + train2 + test2 绿窗 |

这能保证模型在 transform 预测行时看不到目标红窗，只能看到合法历史和绿窗。

## 属性聚合

除总流量外，还读取 volume 表中的车辆属性：

- `vehicle_model`
- `has_etc`
- `vehicle_type`

处理方式：

1. 同样按 20 分钟窗口聚合。
2. 按 `(window_start, tollgate_id, direction, attr_name, attr_value)` 计数。
3. 在特征中形成属性计数和占比。

聚合结果类型可以理解为：

```text
AttrKey = (datetime, tollgate_id, direction, attr_name, attr_value)
value = count
```

这些属性只来自已知窗口：

- 训练样本使用训练日对应窗口。
- 验证/预测样本使用题目给出的测试绿窗。

如果某个预测绿窗缺少某个属性取值，特征中对应计数为 0 或缺省，不会从目标红窗补值。

## 特征构造

四个模型共享基础特征，主要由 `FeatureBuilder` 生成：

- tollgate / direction / combo one-hot
- target slot one-hot
- weekday / weekend / day_of_month
- 目标时间的 `sin/cos`
- 同日绿窗 6 个 20 分钟流量
- 绿窗 `sum`, `mean`, `std`, `trend`
- 车辆属性计数和占比
- `lag_1`, `lag_7`
- combo 均值、combo-slot 均值
- 历史滚动均值和中位数
- 国庆相关标记

之后使用 `DEFAULT_DROP_FEATURES` 剪掉在验证中不稳定的特征。

特征构造有两个阶段：

1. `fit_stats(train_rows)` 只基于训练目标行拟合历史统计。
2. `transform(pred_rows, known_agg, known_attr_agg)` 用已知世界生成预测特征。

因此，预测行的统计特征来自训练期和合法绿窗，不会从待预测红窗回填。

## 样本行和提交行

训练或验证目标行由 `make_target_rows(days, combos)` 生成：

```text
days * TARGET_TIMES * combos
```

phase2 提交行由 `make_target_rows_like_sample(sample_path, first_pred_day)` 生成。它读取 `submission_sample_volume.csv` 的行结构，然后把样例日期整体平移到 test2 第一预测日。

这样做有两个目的：

- 保持提交文件行顺序和样例一致。
- 避免评分脚本按位置检查时出现日期、combo 或 time window 顺序错配。

## `validate-ensemble` 的数据使用

目标：无泄露评估 Oct.18-Oct.24。

数据流：

1. train1 全部日期为 Sep.19-Oct.17。
2. 从 train1 中拆出最后 7 天作为校准集。
3. 校准集训练部分只使用最后 7 天之前的数据。
4. 校准集预测输入只加入校准日同日绿窗。
5. 校准权重只由 train1 内部得到，并按目标小时分别学习。
6. phase1 验证时，使用完整 train1 训练基础模型。
7. phase1 输入只合并 test1 的绿窗数据。
8. 最后用 train2 标签计算验证 MAPE。

更具体地说：

```text
校准权重:
train1 earlier days -> train model
train1 latest 7 days green windows -> prediction features
train1 latest 7 days red windows -> optimize hourly weights

phase1 验证:
train1 all days -> train model
test1 green windows -> prediction features
train2 red windows -> score only
```

关键点：

- train2 标签只在最终评分时使用。
- train2 标签不参与 `validate-ensemble` 权重拟合。
- train2 标签不参与 `validate-ensemble` 基础模型训练。
- train2 标签不参与 `validate-ensemble` 特征统计。

## `predict-ensemble` 的数据使用

目标：生成 Oct.25-Oct.31 phase2 提交。

数据流：

1. 用 train1 训练模型。
2. 用 test1 绿窗预测 Oct.18-Oct.24。
3. 用 train2 标签按目标小时校准融合权重。
4. 用 train1 + train2 训练最终模型。
5. 用 test2 绿窗预测 Oct.25-Oct.31。
6. 按 `submission_sample_volume.csv` 的行顺序写出提交。

更具体地说：

```text
phase2 权重校准:
train1 labels -> train candidate models
test1 green windows -> prediction features
train2 labels -> optimize hourly weights

phase2 最终预测:
train1 + train2 labels -> train candidate models
test2 green windows -> prediction features
no test2 red labels -> only output submission
```

关键点：

- train2 标签在 phase2 场景中是已发布训练数据。
- test2 只提供 Oct.25-Oct.31 的绿窗，不提供红窗标签。
- 输出目标红窗没有被读取或反推。
- `calibration_mape=0.111638` 是历史校准误差，不是目标周真实误差。

## 天气和 trajectory 的使用状态

天气文件会通过 `load_weather` 读取，但当前四模型融合中候选矩阵构造使用：

```text
FeatureBuilder(train_agg, weather, include_weather=False)
```

也就是说，天气数据当前不是 SOTA 融合路线的有效特征来源。保留读取逻辑是为了和主 pipeline 兼容，也方便后续重新打开实验。

trajectory 表当前未参与 SOTA 融合。原因是之前的 trajectory 绿窗统计没有稳定超过当前默认路线。更重要的是，trajectory 很容易误用未来信息：如果按整天或完整红窗后统计轨迹，就会把预测时刻之后的信息带入特征。

后续如果重新使用 trajectory，至少需要满足：

1. 对每个预测目标窗口定义可见截止时间。
2. 只聚合截止时间之前已经发生的轨迹。
3. 在 train1 内部做同样规则的无泄露验证。
4. 不能只看 phase2 校准 MAPE 决定是否采用。

## MAPE 对数据处理的影响

MAPE 让低流量样本更重要，因此当前路线在数据和模型上做了几件事：

- 使用 `log1p(volume)` 缓和大流量的尺度优势。
- 使用 `mape_sample_weight` 提高低流量样本权重。
- 引入 `low_volume_block` 处理最近一周进入低流量 regime 的 combo。
- 引入 `ratio_lag_7`，避免只靠全局均值预测低流量窗口。

这些处理都只使用历史标签或合法绿窗，不依赖目标红窗真实流量。

## 提交文件顺序

提交行不靠内部排序直接生成，而是读取 `submission_sample_volume.csv` 的结构，并把样例日期平移到 phase2 第一预测日。

这样可以避免评分脚本按位置或格式检查时出现 key 顺序错配。

输出写入：

```text
outputs/submission_task2_volume_ensemble.csv
```

每行预测都来自：

```text
target_hour = pred_row.start.hour
pred = prediction_matrix[row] @ weights_by_hour[target_hour]
```

其中 `prediction_matrix` 的行顺序与 `pred_rows` 一致，`pred_rows` 的顺序来自样例提交结构。`target_hour` 是待预测窗口的公开时间字段，例如 `08`, `09`, `17`, `18`，不是目标红窗真实流量。

## 禁止使用的数据

以下用法会构成泄露或不合规：

- 用 Oct.18-Oct.24 标签调 `validate-ensemble` 的权重，再把结果报告成 phase1 无泄露验证分数。
- 用 Oct.18-Oct.24 标签训练 `validate-ensemble` 的基础模型。
- 用 Oct.18-Oct.24 标签拟合 `validate-ensemble` 的 feature statistics。
- 预测 Oct.25-Oct.31 时使用 test2 红窗真实流量。
- 预测某日早高峰时使用该日 `08:00` 之后真实流量。
- 预测某日晚高峰时使用该日 `17:00` 之后真实流量。
- 从完整目标周统计均值、中位数、节假日修正项后再回填到目标周预测。
- 用目标周真实标签选择模型、调参或调融合权重，再报告为无泄露验证。

当前实现没有使用这些数据。

## 数据使用审计清单

修改 SOTA 路线前，建议逐项检查：

1. 新特征是否只来自训练历史或合法绿窗。
2. 新特征的统计量是否只在训练折拟合。
3. `validate-ensemble` 是否仍然只用 train1 内部校准权重。
4. train2 标签是否只在 `validate-ensemble` 最后算分时出现。
5. `predict-ensemble` 是否只把 train2 用作 phase2 已发布训练标签。
6. test2 是否只读取绿窗输入，没有红窗标签来源。
7. 输出行顺序是否仍由 sample 文件控制。
8. 如果新增模型，是否在同一个校准协议下比较，而不是只看目标周后验表现。
