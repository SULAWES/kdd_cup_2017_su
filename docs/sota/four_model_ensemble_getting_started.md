# 四模型融合入门说明

这份文档面向第一次接触本项目的人，目标是说明当前最好方案在做什么、为什么这样做、如何合法复现结果。

## 任务是什么

KDD Cup 2017 Task 2 的目标是预测收费站在未来高峰期的 20 分钟平均车流量。

需要预测的对象有 5 个收费站/方向组合：

- `1_0`
- `1_1`
- `2_0`
- `3_0`
- `3_1`

每天需要预测 12 个时间窗口：

- 早高峰：`08:00-10:00`，共 6 个 20 分钟窗口
- 晚高峰：`17:00-19:00`，共 6 个 20 分钟窗口

一周提交行数：

```text
7 天 * 12 个窗口 * 5 个组合 = 420 行
```

评价指标是 MAPE。它会放大低流量样本的相对误差，所以模型不能只追求普通均方误差低。

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

## 当前最好方案是什么

当前最好方案是四模型融合：

```sh
python run_task2.py validate-ensemble
python run_task2.py predict-ensemble
```

它不是只训练一个模型，而是训练 4 个预测器，然后学习一个非负加权平均：

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
- 用校准集上的 MAPE 最小化来拟合

## 四个模型分别负责什么

### `low_volume_block`

这是主力模型。

它基于 ExtraTrees，并加入一个结构判断：如果某个收费站/方向组合最近 7 天明显低于整体均值和自身历史均值，就改用早晚块模型预测该组合。

当前数据下，这个规则会选择 `1_0`。

直觉上，`1_0` 在训练末期出现低位 regime，用普通全局模型会被其他组合的正常水平拉高；单独切换到 block 模型能降低这部分误差。

### `xgb`

这是 XGBoost 模型。它使用同一批基础特征，但模型族不同，误差形态与 ExtraTrees 不完全一样。

单独看它分数不如主模型，但融合时可能补充一部分主模型的偏差。

### `mlp`

这是一个标准化后的浅层神经网络。

结构是：

```text
StandardScaler -> MLPRegressor(48, 24)
```

它同样不是最强单模型，但提供了非树模型的预测形态。

### `ratio_lag_7`

这个模型预测的不是流量本身，而是相对 7 天前同一窗口的比例：

```text
target = log((volume + 1) / max(lag_7, 1))
```

预测时再乘回 `lag_7`：

```text
pred = exp(model_output) * max(lag_7, 1) - 1
```

它的作用是显式引入周周期。

## 两个命令的区别

### `validate-ensemble`

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

这是可报告的无泄露 phase1 验证结果。

### `predict-ensemble`

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

## 继续改进可以从哪里下手

优先方向：

1. 做更稳健的权重校准，不只依赖最后一周。
2. 给不同 combo 学不同融合权重，但必须用训练折估计。
3. 对 `ratio_lag_7` 加入更稳定的基线，例如 combo-slot 历史中位数。
4. 分析 `1_0` 低位 regime 是否有更明确的触发特征。
5. 如果引入 trajectory 表，必须只使用预测时间之前可见的数据，并先做严格验证。
