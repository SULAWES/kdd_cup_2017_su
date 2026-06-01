# KDD Cup 2017 Task 2 Baseline

目标：预测收费站/方向在指定 20 分钟窗口内的平均 traffic volume。

当前实现先保证端到端可运行，并把数据、特征、模型、流程拆开，方便后续替换更强模型。

## 数据假设

数据位于 `dataset/`，使用本仓库已有文件：

- `dataSets/training/volume(table 6)_training.csv`
- `dataSet_phase2/volume(table 6)_training2.csv`
- `dataSets/testing_phase1/volume(table 6)_test1.csv`
- `dataSet_phase2/volume(table 6)_test2.csv`
- `submission_sample_volume.csv`
- 天气文件 `weather (table 7)_*.csv`

Task 2 的提交格式为 420 行：

- 7 天
- 每天 08:00-10:00 和 17:00-19:00，共 12 个 20 分钟窗口
- 5 个收费站/方向组合：`1_0`, `1_1`, `2_0`, `3_0`, `3_1`

测试流量文件提供同日 06:00-08:00 和 15:00-17:00 的观测窗口，作为目标窗口预测特征。

## 运行

默认模型使用 `scikit-learn` 的 `ExtraTreesRegressor`，目标值默认做 `log1p` 变换。代码仍保留无外部依赖的 `numpy` 岭回归 fallback。

```sh
python run_task2.py validate
python run_task2.py predict
python run_task2.py validate-ensemble
python run_task2.py predict-ensemble
```

可切换模型：

```sh
python run_task2.py --model ridge validate
python run_task2.py --model lgbm validate
python run_task2.py --model hgb validate
python run_task2.py --model extra validate
python run_task2.py --group global validate
python run_task2.py --group block --target-transform raw validate
python run_task2.py --use-weather validate
python run_task2.py --sample-weight-power 0 validate
python run_task2.py --history-blend 0.195 --prediction-scale 0.962 validate
python run_task2.py --no-prune-features validate
```

输出：

- 验证预测：`outputs/validation_phase1_pred.csv`
- phase2 提交：`outputs/submission_task2_volume.csv`

当前 phase1 验证方式：

- 训练：`2016-09-19` 至 `2016-10-17`
- 验证：用 phase1 test 观测窗口预测 `2016-10-18` 至 `2016-10-24`
- 默认 `extra + low_volume_block + log + 观测结构特征（保留观测窗口波动 obs_std）+ 无天气 + 轻量 MAPE 权重 + 剪枝噪声特征` 验证 MAPE：约 `0.120175`
  - 本版在 ExtraTrees 默认超参 `random_state=13, max_depth=14, min_samples_leaf=10` 基础上，增加 recent-low-volume 结构切换：若某 combo 最近 7 天均值同时低于最近整体均值和自身全历史均值的 60%，则该 combo 使用 block 模型，其余使用 global 模型。phase1 和 phase2 训练数据下均只选择 `1_0`。
  - 单纯 `--group global` 为约 `0.120773`；旧默认约 `0.122050`。
- `--history-blend 0.09` 可在 phase1 验证上得到约 `0.119564`，但该权重来自 phase1 验证周调参；训练期滚动周没有支持把它作为默认配置。
- 四模型融合命令：
  - `validate-ensemble`：只用训练期最后一周校准融合权重，再评估 phase1，MAPE 约 `0.118018`，不使用 phase1 验证标签调权。
  - `predict-ensemble`：用已发布的 Oct.18-24 标签校准权重，再预测 Oct.25-31；校准 MAPE 约 `0.116116`，该数合法用于 phase2 权重估计，但不能当作无泄露 phase1 验证分数。
- 递推使用前序目标窗预测、trajectory 绿窗统计、天气特征、分组建模和恢复已剪枝特征均已复测，当前默认下没有带来叠加收益。
- 旧实验 `--history-blend 0.195 --prediction-scale 0.962 --sample-weight-power 0.22` 在旧模型上可得到约 `0.117796`，但这组参数来自该验证集调参，不能作为无泄露默认配置。
- `extra + global + log + 观测结构特征 + 无天气 + 轻量 MAPE 权重 + 不剪枝` 验证 MAPE：约 `0.124342`
- `extra + global + log + 观测结构特征 + 无天气 + 无权重` 验证 MAPE：约 `0.128240`
- `extra + global + log + 观测结构特征 + 天气 + 轻量 MAPE 权重` 验证 MAPE：约 `0.125849`
- `extra + block + raw` 验证 MAPE：约 `0.137534`
- `lgbm + global + log` 验证 MAPE：约 `0.147154`
- `ridge + global + raw` 验证 MAPE：约 `0.196292`

提交文件按 `submission_sample_volume.csv` 的行顺序生成，只把样例日期平移到 phase2 预测日期，避免位置式评分或检查脚本错配 key。

## 架构

- `src/kddcup2017_task2/data.py`：CSV 读取、20 分钟聚合、目标窗口、提交文件生成
- `src/kddcup2017_task2/features.py`：日历、可选天气、观测窗口、车型/ETC 结构、历史统计特征
- `src/kddcup2017_task2/model.py`：模型工厂、岭回归 fallback、树模型和 MAPE
- `src/kddcup2017_task2/pipeline.py`：`validate` / `predict` 流程

## 后续提升方向

优先级建议：

1. 做按 combo 和 target slot 的独立模型，减少不同收费站方向之间的分布干扰。
2. 增加节假日、调休日、工作日类型，以及国庆后恢复期的特殊标记。
3. 使用递推预测：对 08:20 之后、17:20 之后的窗口引入前一目标窗口的预测值。
4. 做更稳健的滚动交叉验证；`history_blend` 和 `prediction_scale` 必须只用训练折拟合，避免把 phase1 验证标签信息固化到默认参数。
5. 增加模型融合：历史规则模型、岭回归、树模型分别产出结果后加权。
6. 如果安装 `pandas`，可以增加更方便的离线分析脚本，但核心训练流程不依赖它。
