# 路线：当前已得到方案整理

本文档把截至目前已经跑过的路线按“可报告正式路线、可继续推进候选、仅作参考路线、低优先级路线”整理，方便后续 Agent 不重复走弯路。

## 结论先行

当前不晋升任何探索路线到 `src/`。正式 SOTA 仍是 `src/` 中的四模型小时权重融合：

```powershell
.\.venv\Scripts\python.exe run_task2.py validate-ensemble
```

无泄露 phase1 MAPE 约 `0.116167`。

最值得继续推进的是“同日绿色观察窗强弱后验校正”。它在 `src1` 中已经表现出更低 phase1 分数和更好的 train1 rolling 分数，但还需要整理成可复现、可解释、选择过程完全不依赖 train2 标签的正式协议后，才能考虑迁入 `src/`。

## 路线状态总表

| 路线 | 代表结果 | 数据合规状态 | 当前状态 |
| --- | ---: | --- | --- |
| ExtraTrees global | `0.120773` | 无泄露 phase1 | 稳定基线 |
| ExtraTrees low_volume_block | `0.120175` | 无泄露 phase1 | 当前最好单模型 |
| 四模型 global 融合 | `0.118018` | 无泄露 phase1 | 上一版 SOTA，可复现对照 |
| 四模型 hour 融合 | `0.116167` | 无泄露 phase1 | 当前正式 SOTA |
| `predict-ensemble` phase2 校准 | `0.111638` | 对 phase2 合法，不是 phase1 无泄露分数 | 只用于 phase2 提交思路 |
| 轨迹第五候选，block cap `0.15` | `0.115924` | phase1 评分无泄露，但 cap 来自探索 | 暂不晋升 |
| 观察窗后验校正，phase1 直选最佳 | `0.114456` | 分数很强，但配置由 phase1 sweep 选出 | exploratory 上界 |
| 观察窗后验校正，rolling 支持配置 | 约 `0.11583` | 选择更接近 train1-only 协议 | 最值得继续验证 |
| 神经先验门控融合 | `0.114758` | phase1 探索结果，seed 敏感，尚无 rolling 协议 | 值得继续验证，暂不晋升 |
| 五节点 PyTorch GNN | `0.133801` | 无泄露 phase1 | 显著对照路线：和主路线差距大，但在直接神经路线中相对较好 |
| 图卷积特征 + ExtraTrees | `0.121563` | 无泄露 phase1 | 弱于单模型，暂停 |
| `src2` Transformer 直接序列预测 | `0.191686` | 无泄露 phase1 评分；初始 CPU 探索 | 发散对照基线，暂不晋升 |
| `src2` LSTM 直接序列预测 | `0.193614` | 无泄露 phase1 评分；初始 CPU 探索 | 发散对照基线，暂不晋升 |
| direct LGBM/HGB/XGB global | `0.159851` 级别 | 无泄露 phase1 | 单独模型弱，仅可作融合多样性参考 |
| history_blend | `0.119564` | 权重来自 phase1 调参 | 只作上界参考 |

## 正式可报告路线

### 1. ExtraTrees Global

用途：稳定基线，帮助判断新特征或新模型是否真的有用。

核心：

- 20 分钟粒度聚合。
- 全局 ExtraTrees 模型。
- `log1p(volume)` 目标。
- 观察窗、车辆结构、历史 lag、日历特征。
- 轻量 MAPE 样本权重。

结果：phase1 MAPE 约 `0.120773`。

说明文档：`docs/route_extra_global.md`。

### 2. Low-Volume Block

用途：当前最好单模型，也是四模型融合中的主力候选。

核心：

- 同时训练 global 和 block 模型。
- 如果某 combo 最近 7 天进入明显低位，则该 combo 切到 block 模型。
- 当前 phase1 / phase2 训练口径下主要触发 `1_0`。

结果：phase1 MAPE 约 `0.120175`。

说明文档：`docs/route_low_volume_block.md`。

### 3. 四模型 Hour 融合

用途：当前正式 SOTA。

候选：

1. `low_volume_block`
2. `xgb`
3. `mlp`
4. `ratio_lag_7`

核心变化：从全局权重改成按目标小时学习融合权重。train1 rolling mean 支持 hour 权重略优，因此比直接按 phase1 挑更激进的 block shrinkage 更有说服力。

结果：phase1 MAPE 约 `0.116167`。

说明文档：

- `docs/route_four_model_ensemble.md`
- `docs/sota/four_model_ensemble_getting_started.md`
- `docs/sota/four_model_ensemble_detailed.md`
- `docs/sota/four_model_ensemble_data_usage.md`

## 暂不晋升但值得继续推进的候选

### 1. 轨迹作为第五融合候选

实现位置：

- `run_task2_traj_exp.py`
- `run_task2_traj_ensemble_exp.py`
- `run_task2_traj_rolling_exp.py`
- `src1/kddcup2017_task2_exp/trajectory_exp.py`
- `src1/kddcup2017_task2_exp/trajectory_ensemble_exp.py`
- `src1/kddcup2017_task2_exp/trajectory_rolling_exp.py`

核心：

- 从 trajectory 表中构造合法绿色窗口统计。
- 作为一个额外候选模型加入四模型融合。
- 对第五候选设置 cap，避免校准折过度相信轨迹候选。

代表结果：

| 配置 | phase1 MAPE |
| --- | ---: |
| block weights + trajectory cap `0.15`，不含 route mean | `0.115924` |
| block weights + trajectory cap `0.10`，不含 route mean | `0.115934` |
| hour weights + trajectory cap `0.10`，不含 route mean | `0.116021` |

rolling 观察：

- `cap=0` 回到四模型 block 基线，folds `0.303073`, `0.143124`。
- `cap=0.20` 主要改善第二折，folds `0.303073`, `0.141428`。
- 轨迹信号有用，但不是稳定主力。

当前判断：暂不晋升。可以作为观察窗后验校正或后续融合候选的一部分继续保留。

### 2. 同日绿色观察窗强弱后验校正

实现位置：

- `run_task2_obs_adjust_exp.py`
- `run_task2_obs_adjust_rolling_exp.py`
- `src1/kddcup2017_task2_exp/observation_adjust_exp.py`
- `src1/kddcup2017_task2_exp/observation_adjust_rolling_exp.py`

核心公式：

```text
adjusted_prediction = base_prediction * exp(beta * log((current_obs_sum + smoothing) / (historical_expected_obs_sum + smoothing)))
```

含义：

- 如果当天绿色窗口比历史同类窗口更强，轻微上调红窗预测。
- 如果当天绿色窗口比历史同类窗口更弱，轻微下调红窗预测。
- `beta_max` 限制校正强度，避免这个后验信号覆盖主模型。

代表结果：

| 配置 | phase1 MAPE | 选择依据 |
| --- | ---: | --- |
| `traj_hour_cap010` + `(combo, block)` expected obs + hour beta，`beta_max=0.10` | `0.114456` | phase1 直选最佳 |
| `hour4` + `(combo, block)` expected obs + hour beta，`beta_max=0.10` | `0.114607` | 不依赖轨迹第五候选 |
| `traj_block_cap020` + `(combo, block, dow)` expected obs + block beta | 约 `0.11583` | rolling 支持更强 |

rolling 支持情况：

| beta cap | rolling 最佳路线 | fold scores | mean MAPE |
| ---: | --- | --- | ---: |
| `0.05` | `traj_block_cap020` + combo/block/dow expected obs + block beta | `0.295705`, `0.140619` | `0.218162` |
| `0.10` | 同上 | `0.290683`, `0.140061` | `0.215372` |
| `0.15` | 同上 | `0.287673`, `0.139862` | `0.213768` |
| `0.20` | 同上 | `0.286962`, `0.139852` | `0.213407` |

当前判断：这是下一步最有价值路线，但仍不晋升到 `src/`。下一步应固定一个 train1-only 选择规则，例如：

1. 只用 train1 rolling 选择 base、expected obs 粒度、adjustment scope、beta cap。
2. 用该选择结果再跑 phase1，仅作为最终评分。
3. 如果文档和代码都能说明没有 train2 标签参与选择，再考虑迁入 `src/`。

### 3. 神经先验门控融合

实现位置：

- `run_task2_torch_nn_exp.py`
- `src1/kddcup2017_task2_exp/torch_nn_exp.py`

核心：

- 不让神经网络直接预测流量。
- 先使用已经表现较好的四模型或五模型融合权重作为 prior。
- 小型 MLP 根据候选预测、观察窗强弱、目标小时、block、combo 等上下文，输出 bounded residual logits。
- 最终权重形式为：

```text
weights = softmax(log(prior_weights) + gate_scale * tanh(MLP(context)))
```

代表结果：

| 配置 | phase1 MAPE |
| --- | ---: |
| `traj_block_cap020` prior，hidden `16`，dropout `0.0`，gate scale `0.40`，seed `13` | `0.114758` |
| 同 base，gate scale `0.30`，seed `13` | `0.114983` |
| `traj_hour_cap010` prior，hidden `16`，dropout `0.1`，gate scale `0.10`，seed `13` | `0.115677` |

seed 稳定性检查：

| Seed | phase1 MAPE |
| ---: | ---: |
| `7` | `0.115089` |
| `13` | `0.114758` |
| `21` | `0.116207` |
| `42` | `0.115718` |

当前判断：

- 这是目前最好的神经网络路线，明显强于直接神经预测和 GCN。
- 它仍然是 phase1 探索结果，且 seed 敏感。
- 暂不晋升。下一步应补 train1-only rolling 版本，并缓存候选矩阵以减少重复训练成本。

## 显著对照路线：PyTorch GNN

这条路线需要在讲解和后续交接中更明确地保留。它不是当前主线，也不接近当前 SOTA，但它是“尝试图神经网络方向后得到的相对较好结果”，能说明为什么最终没有把 GNN 作为核心方案。

实现位置：

- `run_task2_torch_graph_exp.py`
- `src1/kddcup2017_task2_exp/torch_gcn.py`

核心做法：

- 将五个 `(tollgate_id, direction)` 组合视为图节点。
- 每个目标时间窗形成一个小图样本。
- 节点特征来自当前 tabular feature builder 的合法特征。
- 模型使用可学习 node embedding，加上 self/neighbor message passing、LayerNorm 和 Dropout。
- 邻接矩阵尝试了 `identity`、`topology`、`corr` 和 `full`。

代表结果：

| 配置 | Internal MAPE | Phase1 MAPE |
| --- | ---: | ---: |
| `full`, hidden `64`, dropout `0.0`, lr `0.003` | `0.147408` | `0.133801` |
| `topology`, hidden `64`, dropout `0.0`, lr `0.003` | `0.150532` | `0.135571` |
| `corr`, hidden `128`, dropout `0.1`, lr `0.001` | `0.153298` | `0.136018` |
| `identity`, hidden `64`, dropout `0.1`, lr `0.003` | `0.147753` | `0.142225` |

对照意义：

- 相比 numpy pure GCN 的 `0.172921`，PyTorch GNN 明显更好，说明方向不是完全无效。
- 相比 direct tabular/sequence NN 中大量 `0.145+` 或更差的结果，PyTorch GNN 属于较好的直接神经网络路线。
- 但它仍明显弱于当前正式四模型 hour 融合 `0.116167`，也弱于单模型 ExtraTrees `0.120175`。
- 主要瓶颈不是 PyTorch 实现质量，而是图本身只有五个节点，人工拓扑关系太弱，message passing 容易抹掉收费站/方向的个体差异。

当前判断：

- 保留为“GNN 方向的最佳直接模型对照”。
- 不作为短期冲分主线。
- 如果未来继续图方向，应优先构造更有信息量且合规的 route/trajectory 上游关系图，而不是继续在五节点 tollgate graph 上调参。

## 已尝试但低优先级路线

### 1. 其他五节点 GCN / 图卷积

实现位置：

- `run_task2_graph_exp.py`
- `run_task2_torch_graph_exp.py`
- `run_task2_torch_meta_exp.py`
- `src1/kddcup2017_task2_exp/graph_gcn.py`
- `src1/kddcup2017_task2_exp/torch_gcn.py`
- `src1/kddcup2017_task2_exp/torch_meta_ensemble.py`

结果：

| 路线 | 最好 phase1 MAPE |
| --- | ---: |
| numpy pure GCN | `0.172921` |
| graph-convolved features + ExtraTrees | `0.121563` |
| PyTorch graph meta-ensemble | `0.125859` |

判断：除 PyTorch GNN 可作为显著对照路线保留外，其他五节点图路线目前收益有限。当前五个 tollgate-direction 节点太少，人工图结构弱，图平滑容易抹掉 combo 差异。除非引入更丰富且合法的 route/trajectory 上游关系图，否则不建议继续作为主线。

### 2. 直接神经网络预测

实现位置：

- `run_task2_torch_nn_exp.py`
- `src1/kddcup2017_task2_exp/torch_nn_exp.py`
- `run_task2_src2_nn_exp.py`
- `src2/kddcup2017_task2_exp2/sequence_nn_exp.py`

代表结果：

| 路线 | 最好 phase1 MAPE |
| --- | ---: |
| tabular MLP / ResNet 直接预测 | `0.145551` |
| sequence Conv1D 直接预测 | `0.164807` |
| sequence GRU 直接预测 | `0.282066` 级别 |
| `src2` Transformer 直接序列预测 | `0.191686` |
| `src2` LSTM 直接序列预测 | `0.193614` |
| neural residual calibrator | `0.120547` |

`src2` 的两次新增发散探索使用同一套合法 phase1 边界：训练只用 train1 标签，内部 early stopping 的 holdout 周只暴露同日绿色观察窗，train2 标签只在最后评分。当前 CPU 初始结果如下：

| 模型 | 配置 | Internal MAPE | Phase1 MAPE | 结论 |
| --- | --- | ---: | ---: | --- |
| Transformer | hidden `64`, dropout `0.1`, lr `0.001` | `0.240544` | `0.191686` | 两次 src2 探索中较好，但仍远弱于树模型 |
| LSTM | hidden `32`, dropout `0.0`, lr `0.003` | `0.220117` | `0.193614` | 比 smoke 稳定，但不具备竞争力 |

判断：直接神经网络在该小样本表格/序列问题上泛化较差；`src2` LSTM / Transformer 能跑通，但只是发散对照基线。神经残差校准会改善 calibration MAPE 但伤害 phase1。相比之下，带先验约束的 neural gate 更值得继续。

### 3. Direct Boosting 替换

代表结果：

- LightGBM global：约 `0.159851`
- HGB global：约 `0.161509`
- XGBoost global：约 `0.163907`

判断：直接替换 ExtraTrees 不成立。XGBoost 仍可作为融合候选，因为它和 ExtraTrees 的误差形态不同，但不适合单独作为主模型。

### 4. History Blend

代表结果：

- `--history-blend 0.09`：约 `0.119564`
- 旧组合 `--sample-weight-power 0.22 --history-blend 0.195 --prediction-scale 0.962`：约 `0.117796`

判断：说明周周期残差可被修正，但较好参数来自 phase1 调参，rolling 不支持直接作为默认配置。

## 推荐后续顺序

1. 保持正式 `src/` 不变，先完善观察窗后验校正的 train1-only 选择协议。
2. 将 rolling 支持配置和 phase1 直选配置分开命名，避免混淆。
3. 如果要迁入 `src/`，优先迁入 rolling 支持配置，而不是 `0.114456` 的 phase1 直选配置。
4. 同步更新 `docs/sota/`，明确由四模型 hour 融合到观察窗后验校正的思路链条。
5. 迁入前补一个命令级复现流程：从 clean workspace 安装依赖、运行验证、输出指标。
