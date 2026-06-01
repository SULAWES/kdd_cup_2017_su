# 路线：ExtraTrees global 稳定基线

## 核心思路

这是当前所有路线的基础模型。先把原始过车记录聚合到 20 分钟粒度，再用全局 `ExtraTreesRegressor` 学习 5 个收费站/方向组合和 12 个目标时段的共同规律。

关键设置：

- 模型：`ExtraTreesRegressor`
- 分组：`global`
- 目标：`log1p(volume)`
- 样本权重：`sample_weight_power=0.3`，让训练更贴近 MAPE
- 特征：观测绿窗、车型/ETC/车辆类型结构、日历、历史 lag 和历史统计
- 默认剪枝：移除在验证中表现不稳的噪声特征

## 成绩

命令：

```sh
python run_task2.py --group global validate
```

phase1 MAPE：约 `0.120773`。

旧默认约 `0.122050`。当前改善来自 ExtraTrees 默认参数调整：

- `random_state=13`
- `max_depth=14`
- `min_samples_leaf=10`

训练期滚动周也支持这组设置优于旧默认：

- `2016-09-27` 折：`2.694666 -> 2.636691`
- `2016-10-04` 折：`0.244101 -> 0.235495`
- `2016-10-11` 折：`0.148278 -> 0.147794`

## 泄露检查

该路线只用训练期真实标签和测试日已给出的绿窗观测。预测 Oct.18-24 时没有读取 Oct.18-24 红窗真实标签。

## 局限

`1_0` 组合在 Oct.11-17 后进入低位 regime，global 模型会把它和其他组合混在一起学习，局部误差偏高。这直接引出了 `low_volume_block` 路线。
