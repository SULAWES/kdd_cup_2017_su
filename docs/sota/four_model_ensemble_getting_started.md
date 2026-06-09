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

## 从审题到当前方案的完整路线

这一节按当前代码说明项目是怎么一步步做出来的。核心依据是 `problem.md`、`src/kddcup2017_task2/` 的实现和 `outputs/experiments/` 中的实验输出，不以后写的说明文档反推。

### 1. 审题：先把比赛问题变成监督学习问题

题目要求预测 Task 2 的 tollgate traffic volume。读题后先确定四件事：

| 问题 | 代码中的对应 |
| --- | --- |
| 预测对象是什么 | `TargetRow(tollgate_id, direction, start)` |
| 预测哪些组合 | `1_0`, `1_1`, `2_0`, `3_0`, `3_1` |
| 预测哪些窗口 | `TARGET_TIMES` 中的 12 个红窗 |
| 可以使用哪些输入 | 训练历史 + 同日绿窗 + 更早历史 |

因此一行训练样本不是原始 CSV 的一行车辆记录，而是：

```text
一个 tollgate-direction combo 在某一天某个 20 分钟红窗的流量
```

例如：

```text
TargetRow(tollgate_id=1, direction=0, start=2016-10-25 08:40:00)
```

表示预测 `1_0` 在 `2016-10-25 08:40-09:00` 的 traffic volume。

题目中最容易误读的是测试集。测试 volume 文件不是答案文件，它只给出当天绿窗：

```text
06:00-08:00
15:00-17:00
```

这些绿窗是合法输入。真正要预测的红窗是：

```text
08:00-10:00
17:00-19:00
```

所以整个项目的第一原则是：

```text
预测某个红窗时，只能用这个红窗之前已经可见的信息。
```

### 2. 查找和确认资料：只确认题面约束和数据格式

本项目没有依赖外部排行榜经验来决定默认方案。实际用到的信息主要来自：

- `problem.md`：确认红窗、绿窗、phase1/phase2 数据交换和 MAPE 指标。
- `dataset/submission_sample_volume.csv`：确认提交文件结构和 420 行顺序。
- volume CSV 表头：确认时间、收费站、方向、车辆属性字段。
- 当前代码输出：确认每条路线的本地验证分数和融合权重。

这一步的目的不是找复杂模型，而是先防止数据边界错。KDD Cup 2017 Task 2 的难点主要在数据使用规则：如果把 Oct.18-Oct.24 标签提前用于 phase1 调参，或者把 test2 红窗真实流量混进特征，模型分数会虚高但不合规。

### 3. 数据处理：从车辆流水变成窗口级样本

代码入口是 `run_task2.py`，它只负责把 `src/` 加入路径并调用 `kddcup2017_task2.pipeline.main()`。

数据处理集中在 `data.py`：

1. `read_volume_aggregate()` 读取 volume CSV。
2. `floor_20min()` 把每辆车时间向下取整到 20 分钟窗口。
3. 按 `(window_start, tollgate_id, direction)` 计数。
4. `make_target_rows()` 生成训练或验证目标行。
5. `make_target_rows_like_sample()` 按 sample 文件结构生成 phase2 提交行。

原始数据是一车一行，模型需要的是窗口级标签：

```text
WindowKey = (window_start, tollgate_id, direction)
value = 这个 20 分钟窗口内的车辆数
```

提交行数来自题目结构：

```text
5 combos * 12 target windows * 7 days = 420 rows
```

代码没有自己临时排序提交行，而是读取 `submission_sample_volume.csv` 的行结构并整体平移日期。这样做是为了避免提交文件和样例顺序不一致。

### 4. 数据边界：区分 train_agg 和 known_agg

代码里最重要的设计是把“训练标签”和“预测时可见世界”分开：

```text
train_agg = 有标签、可用于训练模型和拟合统计量的数据
known_agg = 预测时允许看见的数据
```

在 phase1 无泄露验证中：

```text
train_agg = train1
known_agg = train1 + test1 绿窗
score labels = train2，只在最后算 MAPE
```

在 phase2 最终预测中：

```text
train_agg = train1 + train2
known_agg = train1 + train2 + test2 绿窗
score labels = 不存在，本地只能生成提交
```

这解释了为什么代码把 `validate-ensemble` 和 `predict-ensemble` 分成两个命令：

| 命令 | 标签使用 | 是否可报告为 phase1 无泄露验证 |
| --- | --- | --- |
| `validate-ensemble` | train2 只在最后算分 | 是 |
| `predict-ensemble` | train2 用来调 phase2 融合权重 | 否，但对 phase2 合法 |

### 4.1 什么样的数据使用是合规的

判断一份数据能不能用，不看它在文件夹里叫 train 还是 test，而看它在当前预测时刻是否已经公开、是否早于被预测窗口、是否只用于允许的环节。

可以用的数据分为三类：

| 数据类型 | 合规用法 | 例子 |
| --- | --- | --- |
| 已发布训练标签 | 可以训练模型、拟合历史统计、调合法阶段的权重 | train1 可用于 phase1；train1 + train2 可用于 phase2 |
| 测试日绿窗 | 可以作为预测日的输入特征 | 预测 Oct.25 早高峰时使用 Oct.25 `06:00-08:00` |
| 目标行自身信息 | 可以作为特征或分组键 | tollgate、direction、目标小时、weekday |

不可以用的数据也可以直接列清：

| 数据类型 | 不合规原因 | 例子 |
| --- | --- | --- |
| 待预测红窗真实流量 | 这是答案本身 | 预测 Oct.25 `08:40` 时使用 Oct.25 `08:40-09:00` volume |
| 被预测窗口之后的数据 | 未来信息 | 预测 Oct.19 时使用 Oct.20 的流量 |
| phase1 验证标签参与调参 | 会把验证集答案固化进方案 | 用 Oct.18-Oct.24 标签挑 `history_blend` 后报告 phase1 分数 |
| 目标周整体统计 | 间接泄露红窗标签 | 先统计 Oct.25-Oct.31 全周均值再回填预测 |

本项目里可以按场景判断：

| 场景 | 合规训练标签 | 合规预测输入 | 标签能做什么 |
| --- | --- | --- | --- |
| train1 内部校准 | 校准日前的 train1 | 校准日绿窗 | 校准日红窗只用于学习融合权重 |
| phase1 无泄露验证 | 完整 train1 | test1 绿窗 | train2 只允许最后算 MAPE |
| phase2 权重校准 | 完整 train1 | test1 绿窗 | train2 已发布，可以调 phase2 权重 |
| phase2 最终预测 | train1 + train2 | test2 绿窗 | test2 红窗没有本地标签，不能使用 |

几个容易混淆但合规的点：

- `test1` / `test2` 的绿窗可以用，因为题面明确提供它们作为预测红窗前的先导信息。
- `lag_7` 可以用，但只能从 `known_agg` 查已经可见的 7 天前同窗口；如果 7 天前还没发布，就只能回退到历史统计。
- 用目标小时学习融合权重是合规的，因为 `08`, `09`, `17`, `18` 来自目标行时间，不是目标红窗真实流量。
- `predict-ensemble` 用 train2 调权是合规的 phase2 做法，因为预测 Oct.25-Oct.31 时，Oct.18-Oct.24 已经作为新增训练标签发布。

当前代码用几个机制保证合规：

| 机制 | 对应代码 | 作用 |
| --- | --- | --- |
| `known_agg` 和 `train_agg` 分离 | `pipeline.py`, `ensemble.py` | 防止训练统计和预测可见数据混在一起 |
| `observation_windows_only()` | `ensemble.py` | 只把校准日绿窗加入已知世界 |
| `fit_stats(train_rows)` | `features.py` | 历史均值只由训练行拟合 |
| `transform(pred_rows, known_agg, known_attr_agg)` | `features.py` | 预测特征只从已知世界取数 |
| `validate-ensemble` / `predict-ensemble` 分命令 | `pipeline.py` | 区分无泄露验证和 phase2 合法调权 |

可以用一句检查规则总结：

```text
如果某个数据在真实比赛提交的那个时刻还不可见，它就不能进入训练、特征、调参或融合权重选择。
```

### 5. 特征工程：尽量只用稳定、可见的信息

特征构造在 `features.py` 的 `FeatureBuilder`。它分两步：

```text
fit_stats(train_rows)
transform(pred_rows, known_agg, known_attr_agg)
```

`fit_stats()` 只看训练目标行，用来拟合 combo 均值、combo-slot 均值、中位数等历史统计。`transform()` 给目标行生成特征，只能从 `known_agg` 和 `known_attr_agg` 取数。

当前主要特征包括：

| 特征组 | 例子 | 为什么需要 |
| --- | --- | --- |
| 身份特征 | tollgate、direction、combo | 不同收费站方向分布不同 |
| 时间特征 | weekday、slot、hour、sin/cos | 高峰内部也有形状差异 |
| 绿窗流量 | 6 个 obs 窗口、sum、mean、std、trend | 题目给出的当天先导信号 |
| 历史滞后 | `lag_1`, `lag_7` | 日周期和周周期 |
| 历史统计 | combo mean、combo-slot mean | 缺失滞后时的稳健回退 |
| 车辆属性 | model / etc / veh_type 的绿窗计数和占比 | 绿窗结构可能预示红窗结构 |
| 节假日标记 | 国庆、节后 | 处理 2016 年 10 月特殊分布 |

天气文件有读取逻辑，但当前 SOTA 路线默认 `include_weather=False`。原因不是题面不能用天气，而是本地验证中天气没有稳定提升，反而增加噪声。

### 6. 指标决定了模型不能只追求普通误差

评估指标是 MAPE：

```text
mean(abs(actual - pred) / actual)
```

低流量窗口会被放大。例如实际为 10 时预测偏 5，误差贡献是 50%；实际为 100 时同样偏 5，只贡献 5%。因此代码做了三件事：

- 训练目标默认用 `log1p(volume)`，降低大流量样本的尺度优势。
- 使用 `mape_sample_weight`，对低流量样本加权。
- 设计 `low_volume_block`，专门处理最近进入低流量 regime 的 combo。

`mape_sample_weight` 的形式是：

```text
weight = (mean(max(y, 1)) / max(y, 1)) ** 0.3
```

这不是直接优化 MAPE，但会让模型训练阶段更重视低流量样本。

### 7. 基座模型选择：为什么主模型是 ExtraTrees

当前单模型主干是 `ExtraTreesRegressor`：

```text
n_estimators=600
max_depth=14
min_samples_leaf=10
random_state=13
```

选择它不是因为它理论上一定最好，而是因为当前特征和数据规模下它最稳。实验输出显示：

| 单模型路线 | phase1 MAPE | 结论 |
| --- | ---: | --- |
| Ridge | `0.196292` | 线性表达能力不足 |
| XGBoost global | `0.163907` | 单独预测较弱 |
| LightGBM global | `0.159851` | 单独预测较弱 |
| HistGradientBoosting global | `0.161509` | 单独预测较弱 |
| ExtraTrees low_volume_block | `0.120175` | 当前最佳单模型 |

ExtraTrees 适合这里的原因是：样本量不大、特征里有大量离散 one-hot 和历史统计，树模型能自然处理非线性和局部组合；ExtraTrees 又比普通 boosting 更不容易在这个小数据上把局部噪声拟合过头。

### 8. 单模型优化：从 global 到 low_volume_block

基础 `train_and_predict()` 支持多种分组：

```text
global
block
combo
combo_block
combo_slot
low_volume_block
```

正式单模型使用 `low_volume_block`。它的逻辑是：

1. 先训练一个 global ExtraTrees。
2. 再训练一个 morning/evening block ExtraTrees。
3. 用 `select_low_volume_combos()` 找最近 7 天明显低位的 combo。
4. 对这些 combo 使用 block 预测，其余 combo 使用 global 预测。

低位 combo 的判断条件是最近均值同时低于：

```text
最近全局均值 * 0.6
自身历史均值 * 0.6
```

这个方法解决的是局部分布漂移。普通 global 模型会把所有 combo 混在一起；当某个 combo 最近明显变低时，全局模型容易预测偏高。`low_volume_block` 用最近历史决定是否切换，而不是用 phase1 标签选择 combo，所以没有验证集泄露。

单模型调参结果显示，当前参数仍是最稳的：

| 路线 | phase1 MAPE |
| --- | ---: |
| `sota_single_extra` | `0.120175` |
| `extra_lv_ratio07` | `0.120489` |
| `extra_lv_ratio05` | `0.121067` |
| `extra_weight02` | `0.121346` |
| `extra_weight04` | `0.123107` |
| `extra_weather` | `0.123011` |

因此没有继续把天气、极端样本权重或更深树写入默认配置。

### 9. 从单模型到四模型融合

单模型到 `0.120175` 后，再提升主要靠降低残差相关性。当前四模型融合候选是：

| 候选 | 训练目标 | 作用 |
| --- | --- | --- |
| `low_volume_block` | `log1p(volume)` | 主力预测 |
| `xgb` | `log1p(volume)` | 另一类树模型偏差 |
| `mlp` | `log1p(volume)` | 非树模型误差形态 |
| `ratio_lag_7` | 相对 `lag_7` 的比例 | 显式建模周周期 |

融合方式是非负凸组合：

```text
pred = prediction_matrix @ weights
weights >= 0
sum(weights) = 1
objective = calibration MAPE
```

优化器使用 `scipy.optimize.minimize(method="SLSQP")`，从均匀权重和每个单模型独占权重多个起点开始，降低局部解风险。

注意：XGBoost、MLP、ratio_lag_7 单独分数都不强，但它们仍然可能有价值。融合看的是残差互补，不是单模型排行榜。比如晚高峰中，`xgb` 和 `mlp` 会获得明显权重，说明它们在某些窗口能补主模型偏差。

### 10. 从上一版 SOTA 到当前 SOTA

上一版 SOTA 是四模型加一组全局权重：

```sh
python run_task2.py validate-ensemble --weight-scope global
```

结果：

```text
validation_mape=0.118018
```

它的问题是所有目标窗口共用同一组权重，相当于假设早高峰和晚高峰的模型误差结构一样。但实验显示并不是这样：

- 早高峰更依赖 `low_volume_block`。
- 晚高峰中 `xgb`, `mlp`, `ratio_lag_7` 的补充价值更明显。

因此在 `src1/` 中探索了不同融合粒度：

| 路线 | phase1 MAPE | 含义 |
| --- | ---: | --- |
| global weights | `0.118018` | 上一版 SOTA |
| combo weights | `0.117580` | 每个 combo 一组权重，提升有限 |
| block weights | `0.116097` | 早晚高峰分开，phase1 很强 |
| hour weights | `0.116166` | 每个目标小时一组权重 |
| slot weights | `0.116959` | 每个 20 分钟 slot 一组，方差更高 |
| combo_block weights | `0.119297` | 校准好但验证差，过拟合 |

直接看 phase1，`block_shrink14/15` 能到约 `0.116001`。但 shrink 比例来自 phase1 sweep，作为正式方案不够干净。为了避免“看了验证集再挑方案”，又做了 train1 内部 rolling fold 选择：

| 路线 | rolling mean MAPE | phase1 MAPE |
| --- | ---: | ---: |
| hour weights | `0.222422` | `0.116166` |
| global weights | `0.223033` | `0.118018` |
| block weights | `0.223099` | `0.116097` |
| slot weights | `0.223868` | `0.116959` |
| combo weights | `0.290771` | `0.117580` |

最终正式选 `hour weights`，理由是：

- 比上一版 global 更强。
- 由 train1 rolling fold 支持，不是直接按 phase1 后验挑最低分。
- 粒度比 block 更细，能区分 `08/09/17/18`。
- 粒度又比 slot、combo、combo_block 更稳。
- 讲解上也清楚：不同目标小时使用不同融合比例。

当前正式命令：

```sh
python run_task2.py validate-ensemble
```

当前正式结果：

```text
weight_scope=hour
validation_mape=0.116167
```

### 11. 当前结果数据和每条路线的含义

当前主线结果如下：

| 路线 | 命令或来源 | MAPE | 含义 |
| --- | --- | ---: | --- |
| 单模型默认 | `validate` | `0.120175` | 最强单模型，ExtraTrees + low_volume_block |
| 四模型全局权重 | `validate-ensemble --weight-scope global` | `0.118018` | 上一版 SOTA |
| 四模型小时权重 | `validate-ensemble` | `0.116167` | 当前正式可报告 phase1 无泄露结果 |
| phase2 合法校准 | `predict-ensemble` | `0.111638` | 用 train2 调权后预测 Oct.25-Oct.31，只是校准误差 |

`validate-ensemble` 当前小时权重如下：

| 目标小时 | `low_volume_block` | `xgb` | `mlp` | `ratio_lag_7` |
| --- | ---: | ---: | ---: | ---: |
| `08` | `0.857320` | `0.000000` | `0.064316` | `0.078364` |
| `09` | `0.933716` | `0.000000` | `0.000000` | `0.066284` |
| `17` | `0.330386` | `0.301570` | `0.285249` | `0.082795` |
| `18` | `0.309732` | `0.213464` | `0.277083` | `0.199722` |

这组权重的直观解释是：

- `08/09` 更相信 ExtraTrees 主模型。
- `17/18` 明显更分散，说明晚高峰中不同模型的误差互补更强。
- `ratio_lag_7` 在所有小时都有一定价值，尤其在 `18` 点更明显。

`predict-ensemble` 的 phase2 校准权重如下：

| 目标小时 | `low_volume_block` | `xgb` | `mlp` | `ratio_lag_7` |
| --- | ---: | ---: | ---: | ---: |
| `08` | `0.622410` | `0.174285` | `0.039189` | `0.164117` |
| `09` | `0.743852` | `0.009330` | `0.094074` | `0.152744` |
| `17` | `0.625160` | `0.019524` | `0.355316` | `0.000000` |
| `18` | `0.160764` | `0.247434` | `0.116063` | `0.475738` |

这组权重只能用于 phase2，因为它用 Oct.18-Oct.24 已发布标签调权。不能把 `0.111638` 当成 phase1 无泄露验证成绩。

### 12. 为什么不采用其他方向

当前没有采用其他路线，不是因为它们不能跑，而是因为代码和实验结果不支持它们作为默认。

| 方向 | 没采用的原因 |
| --- | --- |
| 直接换成 XGBoost / LightGBM / HGB | 单模型 MAPE 约 `0.160`，明显弱于 ExtraTrees |
| 加天气特征 | 当前验证下从 `0.120175` 变差到约 `0.123011` |
| 加更多 ExtraTrees 变体进融合 | 校准更好但验证变差，说明相关性高、容易过拟合 |
| combo_block 权重 | 校准 MAPE 很低，但 phase1 MAPE `0.119297`，过细分组导致方差高 |
| slot 权重 | phase1 有提升，但 rolling fold 不如 hour，且每组样本更少 |
| block shrink14/15 | phase1 最低约 `0.116001`，但 shrink 比例来自 phase1 sweep，不够合规稳健 |
| history_blend / prediction_scale | 曾在验证集上有效，但属于后验调参风险，不能作为默认 |
| trajectory 特征 | 当前没有进入 SOTA；如果使用，必须严格限定每个预测时刻前可见的轨迹 |

### 13. 过程中遇到的问题和解决方式

不包括运行环境问题，主要技术问题有这些：

| 问题 | 解决方式 |
| --- | --- |
| 测试 volume 文件容易被误解为标签 | 明确只使用 test1/test2 绿窗，红窗标签只在训练发布后可用 |
| phase1 验证容易泄露 train2 | `validate-ensemble` 中 train2 只用于最后算分，不参与训练和调权 |
| phase2 又必须利用 train2 | 单独写 `predict-ensemble`，把 train2 作为 phase2 已发布训练标签 |
| MAPE 对低流量敏感 | log 目标、低流量样本权重、low_volume_block |
| 某些 combo 出现近期低位 regime | 用训练历史最近 7 天触发 block 模型切换 |
| 弱模型单独分数差 | 只在融合中使用，看残差互补而不是单模型排名 |
| phase1 sweep 容易过拟合 | 用 train1 rolling fold 选择正式路线 |
| 提交行顺序可能错 | 按 sample 文件生成提交行，而不是手写排序 |

### 14. 最终选择和后续改进方向

最终选择是：

```text
ExtraTrees low_volume_block 主模型
+ XGBoost / MLP / ratio_lag_7 补充模型
+ 按目标小时学习非负 MAPE 融合权重
```

这个方法是从上一版全局融合权重的问题自然发展来的：全局权重太粗，不能表达 `08`, `09`, `17`, `18` 的误差差异；而过细的 combo 或 slot 权重又容易方差过高。hour 权重在分辨率、稳健性和可解释性之间比较均衡。

后续最值得继续做的是：

1. 做更严格的 rolling / nested CV，让权重粒度、shrink、成员选择都只由 train1 内部决定。
2. 尝试层级融合，例如 hour 权重向 global 权重自动 shrink，而不是人工 sweep shrink 比例。
3. 引入合法 trajectory 绿窗特征，但必须为每个预测窗口定义可见截止时间。
4. 做 combo-hour 残差分析，找系统性偏高或偏低的局部窗口。
5. 重新设计节假日和节后恢复特征，而不是只用简单标记。
6. 尝试更贴近 MAPE 的训练目标或计数模型，但必须沿用同一套无泄露验证协议。

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
