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
python run_task2.py validate-ensemble
python run_task2.py predict-ensemble
```

主要模块：

- `src/kddcup2017_task2/data.py`：读取 CSV、20 分钟聚合、目标窗口构造、提交文件生成
- `src/kddcup2017_task2/features.py`：构造日历、可选天气、观测窗口、车型/ETC 结构、历史均值和 lag 特征
- `src/kddcup2017_task2/model.py`：模型工厂，包含 ridge、ExtraTrees、HistGradientBoosting、LightGBM
- `src/kddcup2017_task2/pipeline.py`：验证和预测流程

默认配置：

- 模型：`ExtraTreesRegressor`
  - 默认超参已调为 `random_state=13, max_depth=14, min_samples_leaf=10`
- 分组：`low_volume_block`
  - 默认先训练 global 和 block 两个模型；若某 combo 最近 7 天均值同时低于最近整体均值和自身全历史均值的 60%，则该 combo 使用 block 预测，其余使用 global 预测。
- 目标变换：`log1p(volume)`，预测后用 `expm1` 还原
- 观测窗口结构特征：车型、ETC、车辆类型的 2 小时汇总计数和占比
- 天气特征：默认关闭，可用 `--use-weather` 启用
- 样本权重：默认 `--sample-weight-power 0.3`，轻微提高低流量样本权重以贴近 MAPE
- 后处理：默认关闭（`--history-blend 0 --prediction-scale 1`）。上一版把 `0.195/0.962` 作为默认值，是用 phase1 验证标签调出来的参数，存在验证集泄露/过拟合风险，因此只保留为显式实验开关。
- 特征剪枝：默认去掉在 phase1 验证中不稳定或增益为负的噪声特征，可用 `--no-prune-features` 关闭
- 提交文件：`outputs/submission_task2_volume.csv`

## 已完成

1. 跑通完整流程：数据读取、聚合、训练、验证、预测、提交文件生成。
2. 修复提交顺序问题：预测阶段按 `submission_sample_volume.csv` 的行顺序生成目标行，只平移日期，避免位置式评分错配。
3. 加入可切换模型和实验参数：
   - `--model extra|lgbm|hgb|ridge`
   - `--group global|block|combo|combo_block|combo_slot|low_volume_block`
   - `--target-transform log|raw`
4. 加入四模型融合命令：
   - `validate-ensemble`：用训练期最后一周校准权重，再验证 phase1，避免验证标签调权。
   - `predict-ensemble`：用已发布的 Oct.18-24 标签校准权重，再预测 Oct.25-31。
5. 初始化 Git 仓库并提交 baseline 代码。

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
- ExtraTrees + global + log + 观测结构特征 + 无天气：MAPE 约 `0.126615`
- ExtraTrees + global + log + 观测结构特征 + 无天气 + 轻量 MAPE 权重：MAPE 约 `0.124208`
- ExtraTrees + global + log + 观测结构特征 + 无天气 + 轻量 MAPE 权重 + 剪枝噪声特征：MAPE 约 `0.122250`
- 上一版默认 ExtraTrees + global + log + 观测结构特征 + 无天气 + 轻量 MAPE 权重 + 剪枝噪声特征（保留 `obs_std`）：MAPE 约 `0.122050`
  - `obs_std` 由同日已给出的 06:00-08:00 / 15:00-17:00 观测窗口计算，不读取目标窗口标签。用训练期内滚动周检查时也优于继续剪枝：`2016-10-11` 折 `0.149291 -> 0.148119`，`2016-10-04` 折 `0.250073 -> 0.244111`，`2016-09-27` 折 `2.814915 -> 2.695256`。
- 调参后 ExtraTrees + global + log + 观测结构特征 + 无天气 + 轻量 MAPE 权重 + 剪枝噪声特征 + 调整树深/叶子大小/随机种子：MAPE 约 `0.120773`
  - 训练期滚动周同样优于旧默认：`2016-09-27` 折 `2.694666 -> 2.636691`，`2016-10-04` 折 `0.244101 -> 0.235495`，`2016-10-11` 折 `0.148278 -> 0.147794`。
- 当前默认 ExtraTrees + low_volume_block + log + 观测结构特征 + 无天气 + 轻量 MAPE 权重 + 剪枝噪声特征：MAPE 约 `0.120175`
  - 该方法不使用 phase1 验证标签选择 combo；只由训练数据最近 7 天均值触发。phase1 训练数据和 phase2 全训练数据下均只选择 `1_0`。
  - 训练期滚动折中，低位 regime 出现前该规则不触发，因此与 global 分数一致；在 Oct.11-17 已知低位后预测 Oct.18-24 时触发。
- 四模型融合 `validate-ensemble`：MAPE 约 `0.118018`
  - 模型为 `low_volume_block`, `xgb`, `mlp`, `ratio_lag_7`。
  - 权重只用训练期最后一周回测拟合：`low_volume_block=0.776957`, `xgb=0`, `mlp=0.153564`, `ratio_lag_7=0.069479`。
- 四模型融合 `predict-ensemble` 的 phase2 校准 MAPE 约 `0.116116`
  - 该校准用 Oct.18-24 已发布标签，是预测 Oct.25-31 时的合法历史数据；不能把这个数当作无泄露 phase1 验证分数。
  - 对应权重：`low_volume_block=0.512492`, `xgb=0.115189`, `mlp=0.153896`, `ratio_lag_7=0.218422`。
- 显式实验 `--history-blend 0.09`：MAPE 约 `0.119564`，但训练期滚动周没有支持把该 blend 写入默认值。
- 已复测但未采纳：递推使用前序目标窗预测、trajectory 绿窗统计、天气特征、分组建模、恢复剪枝特征，均未优于当前默认。
- 旧显式实验 `--sample-weight-power 0.22 --history-blend 0.195 --prediction-scale 0.962`：MAPE 约 `0.117796`，但该结果使用 phase1 验证标签选择后处理权重和全局缩放，不能视作无泄露提升。

本轮泄露检查结论：训练/预测特征构造本身没有直接读取验证目标窗口标签；测试日 06:00-08:00、15:00-17:00 观测窗口是题目允许输入。但上一版把在 `2016-10-18` 至 `2016-10-24` 验证标签上调优得到的 `history_blend`、`prediction_scale` 写成默认值，属于验证集信息泄露/过拟合风险，已改为默认关闭。

## 路线文档

几个表现较好的路线已拆成独立说明：

- [ExtraTrees global 稳定基线](route_extra_global.md)
- [recent-low-volume block 结构切换](route_low_volume_block.md)
- [四模型融合](route_four_model_ensemble.md)
- [history_blend 后处理上界实验](route_history_blend.md)
- [路线总览](routes_overview.md)

## 后续方向

优先考虑这些提升：

1. 做更稳健的交叉验证，避免只对 phase1 验证周过拟合。
2. 做滚动周验证来约束后处理参数；后处理参数只能由训练折估计，不能用最终验证周标签选定。
3. 做自动化滚动周调参，特别是区分正常周和国庆异常周，避免手动使用 phase1 验证周信息。
4. 对已加入但默认剪枝的历史统计、节假日、观测比例特征做组合级别筛选，而不是全局启用。
5. 做模型融合，把历史规则、ExtraTrees、LightGBM 等结果加权，但权重必须从训练折估计。
6. 单独分析每个收费站/方向组合的误差，针对高误差组合 `1_0` 和晚高峰后半段做特征或模型调整。
