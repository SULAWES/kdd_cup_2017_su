# 四模型融合的数据处理与数据使用说明

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

当前四模型融合没有使用 trajectory 表。之前试验过 trajectory 绿窗统计，但没有优于当前默认路线。

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

## 绿窗输入

题目允许在预测红窗时使用同日更早的绿窗流量：

- 预测早高峰红窗时使用 `06:00-08:00`
- 预测晚高峰红窗时使用 `15:00-17:00`

代码中对应：

```text
OBS_TIMES["morning"] = 06:00 ... 07:40
OBS_TIMES["evening"] = 15:00 ... 16:40
```

这些绿窗输入在测试文件中给出，是合法输入，不属于泄露。

## 属性聚合

除总流量外，还读取 volume 表中的车辆属性：

- `vehicle_model`
- `has_etc`
- `vehicle_type`

处理方式：

1. 同样按 20 分钟窗口聚合。
2. 按 `(window_start, tollgate_id, direction, attr_name, attr_value)` 计数。
3. 在特征中形成属性计数和占比。

这些属性只来自已知窗口：

- 训练样本使用训练日对应窗口。
- 验证/预测样本使用题目给出的测试绿窗。

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

## `validate-ensemble` 的数据使用

目标：无泄露评估 Oct.18-Oct.24。

数据流：

1. train1 全部日期为 Sep.19-Oct.17。
2. 从 train1 中拆出最后 7 天作为校准集。
3. 校准集训练部分只使用最后 7 天之前的数据。
4. 校准集预测输入只加入校准日同日绿窗。
5. 校准权重只由 train1 内部得到。
6. phase1 验证时，使用完整 train1 训练基础模型。
7. phase1 输入只合并 test1 的绿窗数据。
8. 最后用 train2 标签计算验证 MAPE。

关键点：

- train2 标签只在最终评分时使用。
- train2 标签不参与 `validate-ensemble` 权重拟合。

## `predict-ensemble` 的数据使用

目标：生成 Oct.25-Oct.31 phase2 提交。

数据流：

1. 用 train1 训练模型。
2. 用 test1 绿窗预测 Oct.18-Oct.24。
3. 用 train2 标签校准融合权重。
4. 用 train1 + train2 训练最终模型。
5. 用 test2 绿窗预测 Oct.25-Oct.31。
6. 按 `submission_sample_volume.csv` 的行顺序写出提交。

关键点：

- train2 标签在 phase2 场景中是已发布训练数据。
- test2 只提供 Oct.25-Oct.31 的绿窗，不提供红窗标签。
- 输出目标红窗没有被读取或反推。

## 提交文件顺序

提交行不靠内部排序直接生成，而是读取 `submission_sample_volume.csv` 的结构，并把样例日期平移到 phase2 第一预测日。

这样可以避免评分脚本按位置或格式检查时出现 key 顺序错配。

## 禁止使用的数据

以下用法会构成泄露或不合规：

- 用 Oct.18-Oct.24 标签调 `validate-ensemble` 的权重，再把结果报告成 phase1 无泄露验证分数。
- 预测 Oct.25-Oct.31 时使用 test2 红窗真实流量。
- 预测某日早高峰时使用该日 `08:00` 之后真实流量。
- 预测某日晚高峰时使用该日 `17:00` 之后真实流量。

当前实现没有使用这些数据。
