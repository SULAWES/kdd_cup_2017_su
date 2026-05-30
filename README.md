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
```

可切换模型：

```sh
python run_task2.py --model ridge validate
python run_task2.py --model lgbm validate
python run_task2.py --model hgb validate
python run_task2.py --model extra validate
python run_task2.py --group block --target-transform raw validate
```

输出：

- 验证预测：`outputs/validation_phase1_pred.csv`
- phase2 提交：`outputs/submission_task2_volume.csv`

当前 phase1 验证方式：

- 训练：`2016-09-19` 至 `2016-10-17`
- 验证：用 phase1 test 观测窗口预测 `2016-10-18` 至 `2016-10-24`
- 默认 `extra + global + log` 验证 MAPE：约 `0.130091`
- `extra + block + raw` 验证 MAPE：约 `0.137534`
- `lgbm + global + log` 验证 MAPE：约 `0.147154`
- `ridge + global + raw` 验证 MAPE：约 `0.196292`

提交文件按 `submission_sample_volume.csv` 的行顺序生成，只把样例日期平移到 phase2 预测日期，避免位置式评分或检查脚本错配 key。

## 架构

- `src/kddcup2017_task2/data.py`：CSV 读取、20 分钟聚合、目标窗口、提交文件生成
- `src/kddcup2017_task2/features.py`：日历、天气、观测窗口、历史统计特征
- `src/kddcup2017_task2/model.py`：模型工厂、岭回归 fallback、树模型和 MAPE
- `src/kddcup2017_task2/pipeline.py`：`validate` / `predict` 流程

## 后续提升方向

优先级建议：

1. 做按 combo 和 target slot 的独立模型，减少不同收费站方向之间的分布干扰。
2. 增加节假日、调休日、工作日类型，以及国庆后恢复期的特殊标记。
3. 使用递推预测：对 08:20 之后、17:20 之后的窗口引入前一目标窗口的预测值。
4. 做多时间尺度统计：过去 3/7/14 天同窗口均值、中位数、分位数、同比/环比变化。
5. 增加模型融合：历史规则模型、岭回归、树模型分别产出结果后加权。
6. 如果安装 `pandas`，可以增加更方便的离线分析脚本，但核心训练流程不依赖它。
