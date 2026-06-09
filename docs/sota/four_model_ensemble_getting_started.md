# 四模型融合入门说明

这份文档面向第一次接触本项目的人，目标是说明当前最好方案在做什么、为什么这样做、如何合法复现结果，以及运行输出应该如何解读。

如果只想快速得到结论：

- 无泄露 phase1 验证命令：`python run_task2.py validate-ensemble`
- phase2 提交生成命令：`python run_task2.py predict-ensemble`
- 无泄露 phase1 MAPE：`0.116167`
- phase2 合法校准 MAPE：`0.111638`
- phase2 提交文件：`outputs/submission_task2_volume_ensemble.csv`

`0.111638` 是用已经发布的 Oct.18-Oct.24 标签调融合权重后的历史校准误差，只能说明 phase2 场景下的合法校准效果，不能当作无泄露 phase1 验证分数。

上一版 SOTA 是同样四个模型加一组全局融合权重，phase1 MAPE 为 `0.118018`。当前版本不换数据、不换标签边界，只把融合权重细分为 `08`, `09`, `17`, `18` 四个目标小时，phase1 MAPE 提升到 `0.116167`。

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

## 数据为什么是核心难点

这个任务的模型并不复杂，真正容易出错的是数据边界。

原始 volume 表不是一行一个训练样本，而是一行一辆车。比赛要预测的也不是“下一条车流记录”，而是某个收费站、某个方向、某个 20 分钟窗口内的车辆数。也就是说，本项目必须先把原始流水数据整理成窗口级数据，再把窗口级数据拼成模型样本。

还有一个更容易混淆的地方：测试集 volume 文件并不是要预测的答案，而是题目允许使用的当天提前观测流量。它们只覆盖绿窗，不覆盖目标红窗。

初学者可以先记住三句话：

- 原始 CSV 是车辆流水，模型样本是 20 分钟窗口。
- 训练文件里的红窗可以当标签，测试文件里的绿窗只能当输入。
- `validate-ensemble` 和 `predict-ensemble` 最大区别不是模型，而是哪些标签已经合法可用。

## 数据怎么分

本项目按比赛数据交换后的结构理解数据：

| 数据 | 日期 | 文件 | 在本项目中的角色 |
| --- | --- | --- | --- |
| train1 | Sep.19-Oct.17 | `dataset/dataSets/training/volume(table 6)_training.csv` | 初始训练标签和历史特征 |
| test1 | Oct.18-Oct.24 | `dataset/dataSets/testing_phase1/volume(table 6)_test1.csv` | phase1 绿窗输入 |
| train2 | Oct.18-Oct.24 | `dataset/dataSet_phase2/volume(table 6)_training2.csv` | phase2 新增训练标签 |
| test2 | Oct.25-Oct.31 | `dataset/dataSet_phase2/volume(table 6)_test2.csv` | phase2 绿窗输入 |
| sample | 样例提交日期 | `dataset/submission_sample_volume.csv` | 提交行顺序和格式模板 |

同一段日期 Oct.18-Oct.24 会同时出现在 `test1` 和 `train2` 中，但它们含义不同：

- `test1` 是比赛 phase1 时公开的输入，只包含可用绿窗。
- `train2` 是 phase2 时发布的新增训练标签，可以用于预测 Oct.25-Oct.31。

所以：

- 做无泄露 phase1 验证时，不能用 train2 调模型或调权重。
- 做 phase2 提交时，可以用 train2，因为它在 phase2 已经是历史训练数据。

## 从原始 CSV 到 20 分钟窗口

volume 原始表一行代表一辆车通过收费站，核心字段是：

| 字段 | 用途 |
| --- | --- |
| `time` 或 `date_time` | 车辆通过时间 |
| `tollgate_id` 或 `tollgate` | 收费站 |
| `direction` | 方向 |
| `vehicle_model` | 车辆型号属性 |
| `has_etc` | 是否 ETC |
| `vehicle_type` | 车辆类型 |

代码会先做 20 分钟向下取整：

```text
2016-10-18 06:37:12 -> 2016-10-18 06:20:00
2016-10-18 06:40:00 -> 2016-10-18 06:40:00
```

窗口按半开区间理解：

```text
[06:20, 06:40)
```

也就是 `06:37:12` 属于 `06:20` 窗口，恰好 `06:40:00` 属于下一个 `06:40` 窗口。

聚合后的主表可以理解为：

```text
WindowKey = (window_start, tollgate_id, direction)
value = 这个 20 分钟窗口内的车辆数
```

例如：

```text
(2016-10-18 06:20:00, 1, 0) -> 17
```

表示 tollgate `1`、direction `0` 在 `06:20-06:40` 之间有 17 辆车。

## 红窗和绿窗

这里的“绿窗输入”指题目明确给出的当天更早时间段流量：

- 预测早高峰时，可使用 `06:00-08:00`
- 预测晚高峰时，可使用 `15:00-17:00`

红窗是真正要预测的目标：

- 早高峰目标：`08:00-10:00`
- 晚高峰目标：`17:00-19:00`

具体到 20 分钟窗口：

| 块 | 可用绿窗 | 待预测红窗 |
| --- | --- | --- |
| morning | `06:00`, `06:20`, `06:40`, `07:00`, `07:20`, `07:40` | `08:00`, `08:20`, `08:40`, `09:00`, `09:20`, `09:40` |
| evening | `15:00`, `15:20`, `15:40`, `16:00`, `16:20`, `16:40` | `17:00`, `17:20`, `17:40`, `18:00`, `18:20`, `18:40` |

可以把一天里的可用信息理解为：

```text
早高峰预测:
06:00-08:00  已知绿窗
08:00-10:00  待预测红窗

晚高峰预测:
15:00-17:00  已知绿窗
17:00-19:00  待预测红窗
```

这些绿窗数据是合法输入。红窗真实流量才是要预测的目标，不能在预测时使用。

## 一行模型样本是什么

模型不是直接吃原始车辆流水，而是吃 `TargetRow`。一行 `TargetRow` 表示一个要预测的目标窗口：

```text
TargetRow(tollgate_id, direction, start)
```

预测期样本示例：

```text
TargetRow(tollgate_id=1, direction=0, start=2016-10-25 08:40:00)
```

这行样本的含义是：

```text
预测 tollgate 1 / direction 0 在 2016-10-25 08:40-09:00 的车流量
```

这行没有本地标签，因为 Oct.25-Oct.31 是 phase2 要提交预测的目标周。

训练期样本示例：

```text
TargetRow(tollgate_id=1, direction=0, start=2016-10-10 08:40:00)
```

如果这行来自训练期，它的标签是：

```text
y = aggregate[(2016-10-10 08:40:00, 1, 0)]
```

每个目标行会生成几类特征：

| 特征组 | 例子 | 来源 |
| --- | --- | --- |
| 身份特征 | tollgate、direction、combo | 目标行本身 |
| 时间特征 | weekday、weekend、slot、time sin/cos | 目标行本身 |
| 绿窗流量 | 6 个 obs 窗口、obs_sum、obs_mean、obs_trend | 同日合法绿窗 |
| 历史滞后 | `lag_1`, `lag_7` | 已知历史同 combo 同窗口 |
| 历史统计 | combo 均值、combo-slot 均值、中位数、滚动均值 | 训练期标签 |
| 车辆属性 | vehicle_model / has_etc / vehicle_type 的绿窗计数和占比 | 同日合法绿窗 |
| 节假日特征 | 国庆、节后标记 | 目标日期 |

注意：目标红窗真实流量不会出现在这行样本的特征里。

## 训练数据和已知数据要分开

代码里有两个容易混淆的概念：

```text
train_agg = 有标签、可用于训练模型和统计历史均值的数据
known_agg = 生成预测特征时允许看见的数据
```

`known_agg` 通常比 `train_agg` 多一部分测试日绿窗，因为绿窗是合法输入。

| 场景 | `train_agg` | `known_agg` |
| --- | --- | --- |
| train1 内部校准 | 校准训练日前的 train1 标签 | 校准训练日前的 train1 标签 + 校准日绿窗 |
| phase1 无泄露验证 | 完整 train1 标签 | 完整 train1 标签 + test1 绿窗 |
| phase2 权重校准 | 完整 train1 标签 | 完整 train1 标签 + test1 绿窗 |
| phase2 最终预测 | train1 + train2 标签 | train1 + train2 标签 + test2 绿窗 |

这种区分非常重要。模型训练历史统计只能来自 `train_agg`，但预测当天的绿窗特征要从 `known_agg` 里取。

如果把目标红窗也放进 `known_agg`，特征里的 `lag`、滚动统计或 obs 相关特征就可能看见答案，这就是泄露。

## 特征统计如何避免泄露

当前特征构造分两步：

1. `fit_stats(train_rows)`：只用训练目标行拟合历史统计。
2. `transform(pred_rows, known_agg, known_attr_agg)`：对目标行生成特征。

这意味着：

- combo 均值、combo-slot 均值、中位数只来自训练期。
- `lag_1` 和 `lag_7` 只从 `known_agg` 中查找已经可见的历史窗口。
- 同日 6 个绿窗来自题目公开输入。
- 如果某个历史窗口不存在，会回退到 combo-slot 均值等训练期统计，而不是偷看目标红窗。

例如预测 `2016-10-25 08:40` 时：

```text
lag_1 -> 2016-10-24 08:40
lag_7 -> 2016-10-18 08:40
绿窗 -> 2016-10-25 06:00-08:00
标签 -> 不存在，等待模型预测
```

在 phase2 中，Oct.18 已经在 train2 中发布，所以 `lag_7` 可以合法使用；在 phase1 无泄露验证中，Oct.18 的标签只在最终评分时使用，不能提前加入训练或统计。

## 当前最好方案一眼看懂

当前最好方案是四模型融合，并且按目标小时分别学习融合权重。整体链路如下：

```text
CSV 原始流量
  -> 20 分钟窗口聚合
  -> 构造每个 combo / 日期 / 目标窗口的特征行
  -> 训练 4 个候选预测器
  -> 在历史校准集上按目标小时学习非负融合权重
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

从上一版 SOTA 到当前版本，可以用一句话讲清楚：

```text
上一版：所有目标窗口共用一组融合权重。
当前版：早高峰 08/09 点、晚高峰 17/18 点分别学习权重。
```

原因是不同目标小时的误差形态不同。早高峰更依赖 `low_volume_block`，晚高峰中 `xgb`、`mlp` 和 `ratio_lag_7` 能提供更多补充。这个选择来自 train1 内部滚动折，而不是用 Oct.18-Oct.24 的 phase1 标签直接挑结果。

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

从数据处理角度看，它其实做了两次预测。

第一次是在 train1 内部模拟未来一周：

```text
更早 train1 标签
  -> 拟合基础模型和历史统计
train1 最后一周绿窗
  -> 生成校准集特征
train1 最后一周红窗标签
  -> 只用于学习融合权重
```

第二次才是真正验证 Oct.18-Oct.24：

```text
完整 train1 标签
  -> 重新拟合基础模型和历史统计
test1 绿窗
  -> 生成 Oct.18-Oct.24 特征
train2 红窗标签
  -> 只在最后计算 validation_mape
```

这个流程里，train2 标签没有进入：

- 基础模型训练
- 特征统计拟合
- 融合权重优化
- test1 预测特征构造

因此 `validate-ensemble` 可以作为无泄露 phase1 验证。

当前结果：

```text
phase1 MAPE = 0.116167
```

当前权重：

| 目标小时 | `low_volume_block` | `xgb` | `mlp` | `ratio_lag_7` |
| --- | ---: | ---: | ---: | ---: |
| `08` | `0.857320` | `0.000000` | `0.064316` | `0.078364` |
| `09` | `0.933716` | `0.000000` | `0.000000` | `0.066284` |
| `17` | `0.330386` | `0.301570` | `0.285249` | `0.082795` |
| `18` | `0.309732` | `0.213464` | `0.277083` | `0.199722` |

这是可报告的无泄露 phase1 验证结果。上一版全局权重结果可用 `python run_task2.py validate-ensemble --weight-scope global` 复现，MAPE 为 `0.118018`。

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
calibration MAPE = 0.111638
```

当前权重：

| 目标小时 | `low_volume_block` | `xgb` | `mlp` | `ratio_lag_7` |
| --- | ---: | ---: | ---: | ---: |
| `08` | `0.622410` | `0.174285` | `0.039189` | `0.164117` |
| `09` | `0.743852` | `0.009330` | `0.094074` | `0.152744` |
| `17` | `0.625160` | `0.019524` | `0.355316` | `0.000000` |
| `18` | `0.160764` | `0.247434` | `0.116063` | `0.475738` |

这个数不能当作无泄露 phase1 验证分数，因为它用到了 Oct.18-Oct.24 标签调权。但在 phase2 预测 Oct.25-Oct.31 时，Oct.18-Oct.24 已经是公开训练数据，所以这种校准是合法的。

输出文件：

```text
outputs/submission_task2_volume_ensemble.csv
```

从数据处理角度看，它也分成“调权”和“最终预测”两段。

先用 phase2 已发布的 train2 做合法调权：

```text
train1 标签
  -> 拟合基础模型和历史统计
test1 绿窗
  -> 生成 Oct.18-Oct.24 特征
train2 红窗标签
  -> 学习四模型融合权重
```

再预测真正提交周 Oct.25-Oct.31：

```text
train1 + train2 标签
  -> 拟合最终基础模型和历史统计
test2 绿窗
  -> 生成 Oct.25-Oct.31 特征
submission_sample_volume.csv
  -> 决定输出行顺序
```

这里的 train2 使用是合法的，因为 phase2 的目标周是 Oct.25-Oct.31，而 Oct.18-Oct.24 在这个阶段已经发布为训练数据。相反，test2 只允许提供绿窗，不能提供 Oct.25-Oct.31 红窗标签。

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
| `weight_scope=hour` | 表示按目标小时分别学习融合权重 |
| `calibration_rows=420` | 校准集行数，一周 420 行 |
| `calibration_mape` | train1 内部校准误差 |
| `weight_08_*`, `weight_09_*`, `weight_17_*`, `weight_18_*` | 各目标小时下四个候选模型的融合权重 |
| `single_*_mape` | 每个候选模型在 phase1 验证周上的单模型 MAPE |
| `validation_rows=420` | phase1 验证行数 |
| `validation_mape` | 最终无泄露 phase1 MAPE |
| `actual_mean` / `pred_mean` | 验证标签均值和预测均值，用于发现整体偏高或偏低 |

`predict-ensemble` 里比较重要的输出字段：

| 字段 | 含义 |
| --- | --- |
| `calibration=train1_to_train2` | 用 train1 预测 train2，再用 train2 标签调权 |
| `leakage_check=legal_for_phase2_only` | 表示该校准只对 phase2 提交合法 |
| `weight_scope=hour` | 表示 phase2 提交也按目标小时分别学习融合权重 |
| `calibration_rows=420` | Oct.18-Oct.24 校准行数 |
| `calibration_mape` | phase2 合法历史校准误差 |
| `weight_08_*`, `weight_09_*`, `weight_17_*`, `weight_18_*` | 用于 phase2 提交的分小时融合权重 |
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

### 误区 2：`predict-ensemble` 的 0.111638 就是最终公开榜分数

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
