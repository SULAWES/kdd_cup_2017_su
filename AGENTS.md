# AGENTS.md

本文件用于给后续接手本仓库的 Agent 快速建立上下文。修改代码或文档前请先读完本文件，再按需阅读下面的文档索引。

## 当前项目状态

- 题目：KDD Cup 2017 Task 2，预测指定收费站/方向在目标 20 分钟窗口内的平均流量。
- 正式实现目录：`src/kddcup2017_task2/`。
- 正式入口：`run_task2.py`。
- 探索目录：`src1/kddcup2017_task2_exp/`，用于保存未晋升到正式方案的实验。
- 当前正式 SOTA：`src/` 中的四模型融合，默认 `validate-ensemble` 使用 `--weight-scope hour`。
- 当前正式无泄露 phase1 MAPE：约 `0.116167`。
- 上一个正式基线：四模型 global 权重，phase1 MAPE 约 `0.118018`。
- 当前最好单模型：ExtraTrees + low-volume block，phase1 MAPE 约 `0.120175`。

最近探索结论：

- 五节点 GCN/图卷积方向已经试过，当前效果明显弱于树模型融合；最好 PyTorch GNN phase1 MAPE 约 `0.133801`，不建议作为近期主线。
- 轨迹数据作为第五候选有合法增量信号。block 融合 + trajectory cap `0.15` 的 phase1 MAPE 约 `0.115924`，但 train1 rolling 支持不够稳定，暂未晋升。
- 同日绿色观察窗强弱的后验校正最有潜力。phase1 直选最好约 `0.114456`；train1 rolling 支持的保守路线约 `0.11583`。该路线仍在 `src1`，尚未正式迁入 `src`。
- 新增非图神经网络探索：直接 tabular/sequence 网络效果较弱；神经先验门控融合目前最好 phase1 约 `0.114758`，但 seed 敏感且尚无 rolling 选择协议，暂不晋升。
- 当前用户明确要求：先不晋升到 `src`，先整理已得到路线并同步文档。后续 Agent 不应直接把 observation adjustment 或 trajectory route 合并进正式实现，除非用户再次明确要求。

## 必须遵守的数据合规规则

- phase1 无泄露验证：
  - 训练模型只能使用 train1 标签。
  - 可以使用 test1 提供的同日绿色观察窗口作为目标窗口预测特征。
  - train2 标签只能用于最后评分，不能用于训练、调权、选超参或构造特征。
- phase2 预测：
  - 可以使用 train1 + train2 已发布标签训练或校准。
  - 可以使用 test2 绿色观察窗口预测 phase2 目标窗口。
  - `predict-ensemble` 中用 train2 校准权重是合法 phase2 做法，但其校准 MAPE 不能当成无泄露 phase1 指标。
- 任何从 phase1 分数中直接挑出的权重、cap、beta、scale，只能标记为 exploratory，不能直接称为正式 SOTA。
- 要晋升方案，优先要求它能被 train1-only rolling folds 选中或支持。
- 不要把目标红窗的真实流量递给特征工程；目标窗内的真实值只能作为训练标签或验证评分。

## 开发与实验规范

- 正式可复现方案放在 `src/` 和 `run_task2.py`。
- 未验证充分的探索放在 `src1/`，并配套根目录 `run_task2_*_exp.py` 入口。
- 探索结果记录到 `docs/experiments/src1_exploration_log.md`，至少包含命令、指标、数据使用边界和是否可晋升的判断。
- 修改正式 SOTA 前，先确认：
  - phase1 指标优于当前 `0.116167`；
  - 选择过程不使用 train2 标签；
  - rolling 验证或其他 train1-only 选择机制能解释为什么选它；
  - 文档同步更新到 `docs/sota/`。
- 当前环境建议使用 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_task2.py validate-ensemble
```

- 输出目录 `outputs/` 主要是实验产物，不要默认提交。
- 提交前先运行能证明本次变更有效的最小验证命令，例如：

```powershell
.\.venv\Scripts\python.exe -m py_compile <changed_python_files>
.\.venv\Scripts\python.exe run_task2.py validate-ensemble
```

## 代码结构索引

- `run_task2.py`：正式任务入口。
- `run_task2_exp.py`：`src1` 传统实验入口，包含 sweep、ensemble sweep、rolling ensemble sweep 等。
- `run_task2_graph_exp.py`：五节点图/GCN numpy 实验入口。
- `run_task2_torch_graph_exp.py`：PyTorch GNN 实验入口。
- `run_task2_torch_meta_exp.py`：PyTorch 图 meta-ensemble 实验入口。
- `run_task2_torch_nn_exp.py`：非图神经网络探索入口，包含 direct tabular/sequence、神经残差校准和神经先验门控融合。
- `run_task2_traj_exp.py`：轨迹聚合特征单模型实验入口。
- `run_task2_traj_ensemble_exp.py`：轨迹作为第五融合候选的实验入口。
- `run_task2_traj_rolling_exp.py`：轨迹 capped 融合的 train1 rolling 检查入口。
- `run_task2_obs_adjust_exp.py`：同日绿色观察窗后验校正 phase1 实验入口。
- `run_task2_obs_adjust_rolling_exp.py`：同日绿色观察窗后验校正 train1 rolling 检查入口。
- `src/kddcup2017_task2/data.py`：CSV 读取、20 分钟聚合、目标窗口和提交文件生成。
- `src/kddcup2017_task2/features.py`：日历、观察窗、车型/ETC、历史统计等特征。
- `src/kddcup2017_task2/model.py`：模型工厂、MAPE、fallback 回归器。
- `src/kddcup2017_task2/pipeline.py`：正式 `validate` / `predict` 流程。
- `src/kddcup2017_task2/ensemble.py`：正式四模型融合、权重学习、phase1/phase2 ensemble 流程。
- `src1/kddcup2017_task2_exp/experiments.py`：传统探索框架和 rolling sweep。

## 文档索引

- `problem.md`：题面和任务约束。
- `README.md`：项目基本运行方式、当前基线、正式命令和架构说明。
- `docs/task2_notes.md`：早期任务分析和数据理解笔记。
- `docs/routes_overview.md`：几条主要路线的总览。
- `docs/route_exploration_candidates.md`：截至当前的正式路线、探索候选、低优先级路线和后续推进顺序整理。
- `docs/route_low_volume_block.md`：low-volume block 单模型路线解释。
- `docs/route_four_model_ensemble.md`：四模型融合路线解释。
- `docs/route_extra_global.md`：ExtraTrees global 路线解释。
- `docs/route_history_blend.md`：history blend 路线解释及为什么未作为正式默认。
- `docs/sota/four_model_ensemble_detailed.md`：当前正式 SOTA 的详细技术说明。
- `docs/sota/four_model_ensemble_data_usage.md`：当前正式 SOTA 的数据处理和数据使用说明。
- `docs/sota/four_model_ensemble_getting_started.md`：给不了解项目的人看的入门文档，重点解释数据、审题、流程和方案演进。
- `docs/experiments/src1_exploration_log.md`：`src1` 探索日志，包含调参、融合、rolling、GCN、轨迹、观察窗后验校正等实验记录。

## 接手建议

1. 先运行 `git status --short`，确认是否有用户未提交文件；不要回滚或覆盖不属于自己的改动。
2. 读 `README.md` 和 `docs/sota/four_model_ensemble_getting_started.md`，理解正式方案。
3. 读 `docs/routes_overview.md` 和 `docs/route_exploration_candidates.md`，先分清正式 SOTA、探索上界和 rolling 支持候选。
4. 读 `docs/experiments/src1_exploration_log.md` 的最新章节，了解尚未晋升的实验细节和命令。
5. 若继续冲分，优先完善“观察窗后验校正”路线的无泄露选择协议；在用户明确同意前，不要迁入 `src/`。
6. 若要讲解项目，优先使用“审题 -> 合规数据边界 -> ExtraTrees 基线 -> 四模型融合 -> rolling 选择 hour 权重 -> 轨迹第五候选 -> 观察窗后验校正探索”的顺序。
