# Task 2 思路与进度

## 目标

KDD Cup 2017 Task 2 要预测收费站/方向在指定 20 分钟窗口内的平均 traffic volume。

提交目标窗口：

- 每天 `08:00-10:00` 和 `17:00-19:00`
- 每段 6 个 20 分钟窗口，共 12 个窗口
- 7 天、5 个收费站/方向组合，共 420 行预测

测试集给出同日较早的观测窗口：

- 早高峰预测使用 `06:00-08:00`
- 晚高峰预测使用 `15:00-17:00`

当前 pipeline 的核心思路是：先把原始过车记录聚合到 20 分钟粒度，再用观测窗口、车型/ETC 结构、日历和历史统计特征预测目标窗口流量。天气特征保留为可选项，但当前验证中默认关闭更好。

## 当前实现

代码入口：

```sh
python run_task2.py validate
python run_task2.py predict
```

主要模块：

- `src/kddcup2017_task2/data.py`：读取 CSV、20 分钟聚合、目标窗口构造、提交文件生成
- `src/kddcup2017_task2/features.py`：构造日历、可选天气、观测窗口、车型/ETC 结构、历史均值和 lag 特征
- `src/kddcup2017_task2/model.py`：模型工厂，包含 ridge、ExtraTrees、HistGradientBoosting、LightGBM
- `src/kddcup2017_task2/pipeline.py`：验证和预测流程

默认配置：

- 模型：`ExtraTreesRegressor`
- 分组：`global`
- 目标变换：`log1p(volume)`，预测后用 `expm1` 还原
- 观测窗口结构特征：车型、ETC、车辆类型的 2 小时汇总计数和占比
- 天气特征：默认关闭，可用 `--use-weather` 启用
- 提交文件：`outputs/submission_task2_volume.csv`

## 已完成

1. 跑通完整流程：数据读取、聚合、训练、验证、预测、提交文件生成。
2. 修复提交顺序问题：预测阶段按 `submission_sample_volume.csv` 的行顺序生成目标行，只平移日期，避免位置式评分错配。
3. 加入可切换模型和实验参数：
   - `--model extra|lgbm|hgb|ridge`
   - `--group global|block|combo|combo_block|combo_slot`
   - `--target-transform log|raw`
4. 初始化 Git 仓库并提交 baseline 代码。

## 当前验证结果

phase1 离线验证方式：

- 训练：`2016-09-19` 至 `2016-10-17`
- 验证：用 phase1 test 的观测窗口预测 `2016-10-18` 至 `2016-10-24`

已记录结果：

- 初始 ridge baseline：MAPE 约 `0.196292`
- ExtraTrees + raw target：MAPE 约 `0.143860`
- ExtraTrees + block + raw，参数调整后：MAPE 约 `0.137534`
- ExtraTrees + global + log：MAPE 约 `0.130091`
- ExtraTrees + global + log + 观测结构特征 + 天气：MAPE 约 `0.127509`
- 当前默认 ExtraTrees + global + log + 观测结构特征 + 无天气：MAPE 约 `0.126615`

## 后续方向

优先考虑这些提升：

1. 做更稳健的交叉验证，避免只对 phase1 验证周过拟合。
2. 增加节假日、调休日、工作日类型、国庆后恢复期等特征。
3. 设计递推预测，让后续目标窗口使用前序目标窗口的预测结果。
4. 加入更多历史统计，如过去 3/7/14 天同窗口均值、中位数、分位数和趋势。
5. 做模型融合，把历史规则、ExtraTrees、LightGBM 等结果加权。
6. 单独分析每个收费站/方向组合的误差，针对高误差组合做特征或模型调整。
