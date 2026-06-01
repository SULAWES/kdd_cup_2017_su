# Task 2 较优路线总览

本目录把当前表现较好的路线拆开说明。阅读顺序建议如下：

1. [ExtraTrees global 稳定基线](route_extra_global.md)
2. [recent-low-volume block 结构切换](route_low_volume_block.md)
3. [四模型融合](route_four_model_ensemble.md)
4. [history_blend 后处理上界实验](route_history_blend.md)

## 分数对比

| 路线 | phase1 MAPE | 泄露状态 | 当前建议 |
| --- | ---: | --- | --- |
| ExtraTrees global | `0.120773` | 无泄露 | 稳定基线 |
| low_volume_block | `0.120175` | 无泄露 | 默认单模型路线 |
| 四模型融合 `validate-ensemble` | `0.118018` | 无泄露 | 当前最佳可报告验证路线 |
| 四模型融合 `predict-ensemble` 校准 | `0.116116` | 对 phase2 合法；不可当 phase1 无泄露验证分数 | 用于生成 phase2 融合提交 |
| `--history-blend 0.09` | `0.119564` | 使用 phase1 验证标签调权 | 只作上界参考 |

## 泄露边界

题目允许预测某日红窗时使用该日更早的绿窗数据，即 `06:00-08:00` 和 `15:00-17:00`。不允许使用被预测窗口及其之后的真实流量。

因此：

- 用 train1 训练，并用 phase1 test 绿窗预测 Oct.18-24，是合法验证流程。
- 用 Oct.18-24 的真实标签调 phase1 验证参数，是验证集泄露。
- 在 phase2 阶段，Oct.18-24 已作为新增训练数据发布；用它校准权重后预测 Oct.25-31 是合法的。
