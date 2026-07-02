# src3_explore：Task 2 结构诊断探索区

`src3_explore/` 是 KDD Cup 2017 Task 2 的独立探索工作区，用来系统理解任务中的可预测结构、噪声来源、异常机制和模型失效模式。它不替代、不迁入、也不污染正式 `src/`、`src1/`、`src2/` 路线。

## 定位

这个目录不是短期冲分区。默认产物是 CSV、轻量 SVG 图表和 experiment card，输出到 `outputs/src3_explore/`。

运行单个实验：

```powershell
.\.venv\Scripts\python.exe -m src3_explore residual_atlas
```

列出实验：

```powershell
.\.venv\Scripts\python.exe -m src3_explore list
```

运行全部原型：

```powershell
.\.venv\Scripts\python.exe -m src3_explore all
```

部分实验会复用正式四模型候选矩阵，耗时接近 ensemble validation。

如果当前 Python 能导入 `xgboost`，candidate-cache 实验会使用正式四候选定义。若环境缺少 `xgboost`，`src3_explore.common.candidate_cache` 会保留相同候选列名，但用 HistGradientBoosting 作为 `xgb` 槽位的诊断 fallback。fallback 只用于残差结构诊断，不是正式 SOTA 分数。

## 可见性边界

所有实验应通过 `src3_explore.common.visibility` 进入数据：

- train1 rolling 诊断：只用较早 train1 日期训练，held-out train1 日期只暴露同日绿色窗口。
- phase1 诊断：只用 train1 标签训练，test1 绿色窗口作为合法输入。
- train2 标签只能在预测固定后接入，用于 phase1 最终观察和评分。
- phase2 行可见但无标签；绝不能读取 test2 红窗标签。

这里输出的 phase1 数字都是诊断观察，不是 SOTA 声明。不能反复 sweep phase1 后把最优配置当作正式路线选择依据。

## 模块索引

| 区域 | 模块 | 作用 |
| --- | --- | --- |
| diagnostics | `residual_atlas.py` | 缓存候选模型预测，按 date、combo、hour、slot、绿窗强度、ETC 占比、轨迹信号、模型分歧汇总残差。 |
| diagnostics | `model_disagreement.py` | 分析候选模型预测分歧，以及真实值在高分歧时更接近哪个模型。 |
| diagnostics | `green_red_transfer_analysis.py` | 拟合受约束 6x6 green-to-red transfer，聚类 green shape，并导出 ratio surface。 |
| diagnostics | `adversarial_validation.py` | 检测 train1、phase1-visible、phase2-visible 和不同时段之间的分布偏移。 |
| representations | `curve_dictionary.py` | 用 PCA/NMF/dictionary day curve 从绿色窗口重建红窗。 |
| representations | `day_embedding_clustering.py` | 构造 day embedding，标记 weekday/weekend、holiday/post-holiday、low-volume、ETC anomaly、allocation anomaly 等 regime。 |
| mechanisms | `route_arrival_kernel.py` | 研究 route/trajectory lead-lag kernel，不继续五节点 GNN 调参。 |
| mechanisms | `tollgate12_allocation.py` | 分析 `z12 = y1 + y2` 和 `r2 = y2 / (y1 + y2)`，定位 allocation anomaly。 |
| mechanisms | `etc_component_model.py` | 把 ETC、vehicle model、vehicle type 当作生成分量，比较 component-level green→red 关系。 |
| probabilistic | `quantile_baselines.py` | 训练 p10/p50/p90 分位数基线，并评估 coverage 与 failure cases。 |
| probabilistic | `conformal_intervals.py` | 用 train1 calibration residual 构造 conformal interval，并与 ensemble spread 对齐。 |
| explain | `graph_signal_audit.py` | 审计五节点 raw/residual/log-residual 图信号是否稳定。 |
| explain | `gnn_message_passing_damage.py` | 模拟五节点 message passing 对 ensemble anchor 的平滑伤害、over-smoothing 和低流量负迁移。 |
| explain | `graph_randomization_test.py` | 多 seed 比较 topology、random、full、identity 边，验证真实边是否优于随机边。 |
| explain | `route_graph_replacement.py` | 用 route/intersection/tollgate lead-lag 特征替代五节点静态图，比较 residual explanation。 |
| explain | `sequence_permutation_test.py` | 比较正常顺序、打乱顺序、summary-only、raw-sequence tree/linear 和已记录 LSTM/Transformer 基线。 |
| explain | `nn_representation_swap.py` | 比较 NN/tree × raw sequence/engineered features，区分模型族问题和表示问题。 |
| explain | `nn_prediction_collapse.py` | 比较真实、ensemble 和 LSTM/Transformer collapse proxy 的分布、分位数和低/高流量 recall。 |
| explain | `noise_robustness_test.py` | 比较标签 winsorize、holiday removed、低流量权重、log/raw target 对 ExtraTrees/XGB/MLP 及神经基线的影响。 |
| explain | `information_decomposition.py` | 递增加入 combo/hour/slot、lag、rolling、green obs、attr、route、disagreement，量化信息源贡献。 |
| explain | `oracle_ensemble_gap.py` | 计算 candidate oracle、ensemble regret，并诊断 oracle winner 是否可由上下文信号预测。 |

## Experiment card 模板

每个实验都写一个 card，路径为 `outputs/src3_explore/cards/<experiment>.md`。统一字段如下：

- 假设（Hypothesis）：这个实验想验证什么。
- 数据可见性（Data visibility）：是否只使用合法可见信息。
- 最小实现（Prototype）：当前原型做了什么。
- 预期洞察（Expected insight）：即使分数不好也能学到什么。
- 指标（Metrics）：MAPE、signed error、calibration、regime 行为、uncertainty coverage 等。
- 结果（Result）：现象总结，不只写一个分数。
- 下一步（Next）：保留、扩展、归档还是放弃。
- 产物（Artifacts）：CSV、SVG 和明细表路径。

## 最近一次本地运行

运行日期：2026-07-02。

命令：

```powershell
python -m src3_explore all --force-cache
```

运行产物：

- cards：`outputs/src3_explore/cards/`
- CSV 和 SVG：`outputs/src3_explore/{diagnostics,representations,mechanisms,probabilistic}/`
- 候选预测缓存：`outputs/src3_explore/cache/`

环境说明：本次运行使用 `candidate_backend=official_xgboost`，当前系统 Python 可导入 `xgboost 3.3.0`。这些仍是 `src3_explore` 诊断观察，不替代正式 `run_task2.py validate-ensemble` 口径。

关键现象：

- Candidate cache：最新 train1 calibration MAPE 为 `0.136385`；phase1 observation MAPE 为 `0.116640`。
- Residual atlas：最差 phase1 分组集中在 `slot=18:40` MAPE `0.1893`、`combo=1_0` MAPE `0.1640`、`hour=18` MAPE `0.1571`、`slot=18:20` MAPE `0.1554`。
- Model disagreement：真实值最近的候选分布为 `low_volume_block=127`、`mlp=90`、`ratio_lag_7=107`、`xgb=96`。候选多样性是真信号，高分歧失败样本主要集中在低流量 `1_0` 晚高峰和部分 `2_0` 晚高峰。
- Green-red transfer：受约束 6x6 transfer 单独预测较弱，train1 fold MAPE `0.254261`，phase1 MAPE `0.285728`。最差分组为 `1_0/evening`，说明简单线性迁移更适合诊断而不是直接建模。
- Curve dictionary：NMF 是最不差的曲线补全基线，train1 fold MAPE `0.389873`，phase1 MAPE `0.466793`；PCA 和普通 dictionary 更弱。只用 6 个 green slot 补全日曲线过于欠定。
- Day embedding：聚类能分出 holiday/ETC/allocation regime。holiday low-volume cluster 的 ETC share 约 `0.12-0.14`，`r2_allocation` 约 `0.05`；正常或节后 cluster 的 `r2_allocation` 约 `0.58-0.61`。
- Route arrival kernel：最强 raw lead-lag correlation 的 `abs(corr)=0.398081`，对应 `A -> tollgate 2` 的 evening long lag。轨迹有机制信号，但不是简单单调 count 特征。
- Tollgate 1/2 allocation：当前 broad threshold 下，train1 有 `99` 个 allocation flags，phase1 最终观察有 `7` 个。它适合作为 residual join key，不足以直接证明计量错误。
- ETC/component model：component-sum ratio 单独较弱，ETC/model/vehicle type 的 phase1 MAPE 都约 `0.193`；vehicle model 在 train1 fold 上略好。
- Quantile baseline：p10-p90 phase1 coverage 为 `0.761905`，低于名义 80%；wide interval 覆盖好于 narrow interval，说明 uncertainty width 有信号但仍未充分校准。
- Conformal interval：split conformal radius 为 `25.647037`，phase1 coverage `0.961905`，mean width `51.230316`。覆盖偏保守，pooled radius 缺少自适应。
- Adversarial validation：AUC 很高（`0.984094` 到 `1.000000`），但 `day_of_month` 主导特征重要性。需要去除绝对日期特征后再解释为真实交通分布偏移。

## Explanation experiments 运行摘要

运行日期：2026-07-03。

目标：解释为什么五节点 GNN、LSTM、Transformer 在本任务上弱于当前结构化 ensemble，而不是继续调参冲分。

关键现象：

- 图信号审计：log residual 平均绝对节点相关 `0.549115`，但稳定 residual 边只有 `4/10`，说明五节点图有局部相关但不够稳定。
- Message passing damage：identity anchor MAPE `0.116640`，最佳非 identity message passing MAPE `0.129317`，最小节点距离比例 `0.405009`，支持 over-smoothing / negative transfer 解释。
- 随机化检验：topology 平均 MAPE `0.182707`，random 平均 MAPE `0.164563`，真实五节点边没有优于随机边。
- Route graph replacement：五节点静态邻居 residual MAPE `0.339606`，route arrival kernel residual MAPE `0.263472`；route 图更有语义，但仍更适合解释或受限融合。
- Sequence permutation：raw-sequence ExtraTrees 正常顺序 MAPE `0.144971`，summary-only MAPE `0.161596`，而已记录 LSTM/Transformer 仍在 `0.19` 左右；序列顺序不是主要优势来源。
- Representation swap：tree engineered MAPE `0.125141`，NN engineered MAPE `0.176640`，tree raw MAPE `0.144971`；问题不只是 raw sequence 表示，模型族和训练方差也重要。
- Prediction collapse：真实 std `35.437279`，ensemble std `33.388182`，LSTM collapse proxy std `11.685864`；直接序列神经模型有分布塌缩风险。
- Noise robustness：ExtraTrees raw MAPE `0.125077`，最佳扰动 MAPE `0.123205`，MLP raw MAPE `0.176640`；树模型对噪声和目标尺度更稳。
- Information decomposition：只用 combo/hour/slot 的 MAPE `0.348402`，逐步加入结构信息后最佳阶段 MAPE `0.126216`；显式结构信息解释了 ensemble 优势。
- Oracle ensemble gap：ensemble MAPE `0.116640`，candidate oracle MAPE `0.057646`，winner CV accuracy `0.369048`；候选多样性有上界，但 winner 预测仍难，必须用 train1-only 协议。

## 当前判断

真信号：

- 同日绿色观察窗强弱是最强、最合题意的信号；`src1` observation posterior adjustment 仍是最值得继续 formalize 的候选。
- 四模型候选存在真实错误多样性，尤其按 hour 和低流量 regime。
- route/trajectory 有机制增量，但 raw lead-lag 相关性中等，不能直接替代树模型。
- `1_0` 低流量和晚高峰晚 slot 是稳定 failure mode。
- tollgate 1/2 allocation 与车辆 component mix 对解释残差有价值，但 standalone 生成模型不强。

更像噪声或低优先级：

- 直接 LSTM/Transformer 序列模型在 `src2` 可运行但远弱于当前树模型融合。
- 继续五节点 tollgate GNN 调参价值低；已有 PyTorch GNN 是重要对照，不是主线。
- 只靠 6 个 green slot 做 PCA/dictionary 曲线补全过弱。
- naive 6x6 green-red transfer 对 `1_0` evening 和 late slots 太粗。
- component ratio generator 不适合作为独立主模型。

值得继续：

- 固化 observation posterior adjustment 的 train1-only 选择协议。
- 把 residual_atlas 高误差行与 route kernel、allocation、ETC component 和 green shape cluster 做交叉诊断。
- 做去除 `day_of_month` 后的 adversarial validation。
- 把 quantile width、conformal miss 和 model disagreement 合成 train1-only 风险标签。

应归档，除非有新证据：

- 不带更丰富 route/trajectory 结构的五节点 GNN 继续调参。
- phase1 sweep 选出的 cap、beta、gate scale。
- 直接 neural sequence prediction 作为正式路线替代。
- standalone PCA/dictionary reconstruction 或 standalone component-ratio generator。
