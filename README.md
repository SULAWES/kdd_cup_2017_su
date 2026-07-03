# KDD Cup 2017 Task 2 Traffic Volume Forecasting

本仓库实现 KDD Cup 2017 Task 2 的收费站流量预测流程：对指定收费站/方向组合，在每天早晚高峰的 20 分钟目标窗口内预测平均车流量。

当前正式方案位于 `src/kddcup2017_task2/`，入口为 `run_task2.py`。正式默认路线是四模型融合，并按目标小时学习融合权重；该路线是当前仓库可复现的正式 SOTA。

## 当前状态

| 路线 | 命令 | phase1 MAPE | 状态 |
| --- | --- | ---: | --- |
| ExtraTrees low-volume block 单模型 | `run_task2.py validate` | 约 `0.120175` | 当前最好单模型 |
| 四模型 global 权重融合 | `run_task2.py validate-ensemble --weight-scope global` | 约 `0.118018` | 上一版正式基线 |
| 四模型 hour 权重融合 | `run_task2.py validate-ensemble` | 约 `0.116167` | 当前正式 SOTA |
| phase2 合法校准融合 | `run_task2.py predict-ensemble` | 校准约 `0.111638` | 仅用于 phase2 提交；不是 phase1 无泄露指标 |

四模型候选为：

- `low_volume_block`
- `xgb`
- `mlp`
- `ratio_lag_7`

默认融合粒度为 `--weight-scope hour`，即 `08`、`09`、`17`、`18` 四个目标小时分别学习非负凸组合权重。该选择由 train1 rolling 检查支持，不是直接用 phase1 标签挑出的后验最优配置。

## 数据边界

本项目按比赛发布阶段区分训练标签、预测时可见输入和最终评分标签。

| 数据 | 日期 | 文件 | 用途 |
| --- | --- | --- | --- |
| train1 | 2016-09-19 至 2016-10-17 | `dataset/dataSets/training/volume(table 6)_training.csv` | phase1 训练标签与历史统计 |
| test1 | 2016-10-18 至 2016-10-24 | `dataset/dataSets/testing_phase1/volume(table 6)_test1.csv` | phase1 同日绿色观察窗口输入 |
| train2 | 2016-10-18 至 2016-10-24 | `dataset/dataSet_phase2/volume(table 6)_training2.csv` | phase1 最终评分标签；phase2 已发布训练标签 |
| test2 | 2016-10-25 至 2016-10-31 | `dataset/dataSet_phase2/volume(table 6)_test2.csv` | phase2 同日绿色观察窗口输入 |
| sample | 提交模板日期 | `dataset/submission_sample_volume.csv` | 输出行顺序和提交格式 |

预测目标为 5 个收费站/方向组合：

```text
1_0, 1_1, 2_0, 3_0, 3_1
```

每天预测 12 个目标窗口：

```text
08:00, 08:20, 08:40, 09:00, 09:20, 09:40
17:00, 17:20, 17:40, 18:00, 18:20, 18:40
```

测试 volume 文件只提供同日绿色观察窗口：

```text
morning: 06:00-08:00
evening: 15:00-17:00
```

目标红窗真实流量不能进入特征、训练、调参、校准或选择逻辑。phase1 验证中，train2 标签只能在预测固定后用于最终评分。

## 环境

推荐使用仓库本地虚拟环境：

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

依赖见 `requirements.txt`，包括 `numpy`、`scikit-learn`、`scipy`、`xgboost`、`lightgbm`、`torch` 等。

## 正式运行命令

无泄露 phase1 验证：

```bash
./.venv/Scripts/python.exe run_task2.py validate-ensemble
```

生成 phase2 提交文件：

```bash
./.venv/Scripts/python.exe run_task2.py predict-ensemble
```

复现上一版全局权重融合：

```bash
./.venv/Scripts/python.exe run_task2.py validate-ensemble --weight-scope global
```

运行当前最好单模型：

```bash
./.venv/Scripts/python.exe run_task2.py validate
```

生成单模型 phase2 提交：

```bash
./.venv/Scripts/python.exe run_task2.py predict
```

常用单模型对照：

```bash
./.venv/Scripts/python.exe run_task2.py --model ridge validate
./.venv/Scripts/python.exe run_task2.py --model lgbm validate
./.venv/Scripts/python.exe run_task2.py --model hgb validate
./.venv/Scripts/python.exe run_task2.py --model extra --group global validate
./.venv/Scripts/python.exe run_task2.py --group block --target-transform raw validate
./.venv/Scripts/python.exe run_task2.py --use-weather validate
./.venv/Scripts/python.exe run_task2.py --no-prune-features validate
```

主要输出：

| 命令 | 输出 |
| --- | --- |
| `validate` | `outputs/validation_phase1_pred.csv` |
| `predict` | `outputs/submission_task2_volume.csv` |
| `validate-ensemble` | `outputs/validation_phase1_ensemble_pred.csv` |
| `predict-ensemble` | `outputs/submission_task2_volume_ensemble.csv` |

`outputs/` 为实验产物目录，默认不纳入提交。

## 探索区

探索代码与正式路线隔离。除非明确晋升，不应把探索配置迁入 `src/`。

### src1：传统探索与候选方案

`src1/kddcup2017_task2_exp/` 保存尚未晋升到正式路线的实验。

常用入口：

```bash
./.venv/Scripts/python.exe run_task2_exp.py --help
./.venv/Scripts/python.exe run_task2_graph_exp.py --help
./.venv/Scripts/python.exe run_task2_torch_graph_exp.py --help
./.venv/Scripts/python.exe run_task2_torch_meta_exp.py --help
./.venv/Scripts/python.exe run_task2_torch_nn_exp.py --help
./.venv/Scripts/python.exe run_task2_traj_exp.py --help
./.venv/Scripts/python.exe run_task2_traj_ensemble_exp.py --help
./.venv/Scripts/python.exe run_task2_traj_rolling_exp.py --help
./.venv/Scripts/python.exe run_task2_obs_adjust_exp.py --help
./.venv/Scripts/python.exe run_task2_obs_adjust_rolling_exp.py --help
```

关键探索结论：

| 方向 | 最好观察结果 | 当前判断 |
| --- | ---: | --- |
| trajectory 第五候选，block cap `0.15` | 约 `0.115924` | 有增量信号，rolling 支持不够稳定 |
| 观察窗后验校正，phase1 直选最佳 | 约 `0.114456` | exploratory 上界，不能作为正式 SOTA |
| 观察窗后验校正，rolling 支持配置 | 约 `0.11583` | 最值得继续 formalize |
| 神经先验门控融合 | 约 `0.114758` | seed 敏感，尚无 rolling 选择协议 |
| PyTorch 五节点 GNN | 约 `0.133801` | 重要对照路线，不作为主线 |

### src2：直接序列神经网络对照

`src2/` 保存 LSTM / Transformer 直接序列预测探索。

运行示例：

```bash
./.venv/Scripts/python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 16 --epochs 30 --patience 30 --output outputs/experiments/src2_sequence_nn_smoke.csv
```

当前最好初始 CPU 探索结果：

| 模型 | phase1 MAPE | 判断 |
| --- | ---: | --- |
| Transformer | 约 `0.191686` | 可运行对照，明显弱于树模型融合 |
| LSTM | 约 `0.193614` | 可运行对照，明显弱于树模型融合 |

### src3_explore：结构诊断与失效模式分析

`src3_explore/` 用于理解可预测结构、噪声来源、异常机制和模型失效模式。该目录不以短期冲分为目标。

列出实验：

```bash
./.venv/Scripts/python.exe -m src3_explore list
```

运行单个实验：

```bash
./.venv/Scripts/python.exe -m src3_explore residual_atlas
./.venv/Scripts/python.exe -m src3_explore model_disagreement
./.venv/Scripts/python.exe -m src3_explore green_red_transfer
./.venv/Scripts/python.exe -m src3_explore curve_dictionary
./.venv/Scripts/python.exe -m src3_explore day_embedding
./.venv/Scripts/python.exe -m src3_explore route_arrival_kernel
./.venv/Scripts/python.exe -m src3_explore tollgate12_allocation
./.venv/Scripts/python.exe -m src3_explore etc_component_model
./.venv/Scripts/python.exe -m src3_explore quantile_baselines
./.venv/Scripts/python.exe -m src3_explore conformal_intervals
./.venv/Scripts/python.exe -m src3_explore adversarial_validation
```

运行全部原型：

```bash
./.venv/Scripts/python.exe -m src3_explore all --force-cache
```

默认输出：

```text
outputs/src3_explore/
outputs/src3_explore/cards/
outputs/src3_explore/cache/
```

`src3_explore` 最近诊断结论：

- 候选模型存在真实多样性；candidate oracle 远强于固定 ensemble，但 winner 上下文预测仍困难。
- `1_0` 低流量、晚高峰 `18:20` / `18:40` 是稳定 failure mode。
- 绿色观察窗强弱是真信号；naive 6x6 green-red transfer 和 PCA/NMF/dictionary 补全只能作为诊断基线。
- route/trajectory 有机制信号，但 raw lead-lag count 不足以直接替代正式树模型。
- tollgate 1/2 allocation、ETC/component mix 更适合作为 residual join key，而不是独立主模型。
- conformal interval 覆盖偏保守；quantile baseline 覆盖不足，说明不确定性仍需按 regime 校准。
- adversarial validation 会被绝对日期特征主导，解释分布偏移前应先去除 `day_of_month` 等显式日期特征。

详细说明见 `src3_explore/README.md`。

## 代码结构

```text
run_task2.py                         正式入口
src/kddcup2017_task2/                正式实现
src1/kddcup2017_task2_exp/           传统探索
src2/kddcup2017_task2_exp2/          直接序列神经网络探索
src3_explore/                        结构诊断与失效模式分析
tests/                               单元测试
docs/                                路线说明、SOTA 文档和实验日志
dataset/                             比赛数据
outputs/                             本地运行产物
```

正式实现模块：

| 文件 | 职责 |
| --- | --- |
| `src/kddcup2017_task2/data.py` | CSV 读取、20 分钟聚合、目标行和提交文件生成 |
| `src/kddcup2017_task2/features.py` | 日历、观察窗、车辆属性、历史统计特征 |
| `src/kddcup2017_task2/model.py` | 模型工厂、MAPE、fallback 回归器 |
| `src/kddcup2017_task2/pipeline.py` | 单模型 `validate` / `predict` 流程 |
| `src/kddcup2017_task2/ensemble.py` | 四模型候选、权重学习、phase1/phase2 融合流程 |

## 文档

| 文档 | 内容 |
| --- | --- |
| `problem.md` | 题面和任务约束 |
| `docs/sota/four_model_ensemble_getting_started.md` | 当前正式方案入门说明 |
| `docs/sota/four_model_ensemble_detailed.md` | 当前正式 SOTA 技术细节 |
| `docs/sota/four_model_ensemble_data_usage.md` | 当前正式 SOTA 数据边界说明 |
| `docs/routes_overview.md` | 主要路线总览 |
| `docs/route_exploration_candidates.md` | 正式路线、探索候选和归档方向 |
| `docs/experiments/src1_exploration_log.md` | `src1` 实验日志 |
| `docs/experiments/src2_exploration_log.md` | `src2` 实验日志 |
| `src3_explore/README.md` | `src3_explore` 结构诊断说明和最近运行摘要 |

## 提交前检查

根据变更范围选择最小有效验证命令：

```bash
./.venv/Scripts/python.exe -m py_compile path/to/changed_file.py
./.venv/Scripts/python.exe -m unittest tests.test_src3_explore_core
./.venv/Scripts/python.exe run_task2.py validate-ensemble
```

正式 SOTA 相关改动必须额外确认：

- phase1 指标优于当前 `0.116167`。
- 选择过程不使用 train2 标签。
- rolling 验证或其他 train1-only 机制能解释为什么选择该配置。
- `docs/sota/` 和相关实验日志同步更新。
