# Task 2 较优路线总览

本目录把当前表现较好的路线拆开说明。阅读顺序建议如下：

1. [ExtraTrees global 稳定基线](route_extra_global.md)
2. [recent-low-volume block 结构切换](route_low_volume_block.md)
3. [四模型融合](route_four_model_ensemble.md)
4. [history_blend 后处理上界实验](route_history_blend.md)

四模型融合的展开说明：

- [四模型融合入门说明](sota/four_model_ensemble_getting_started.md)
- [四模型融合方法详细说明](sota/four_model_ensemble_detailed.md)
- [四模型融合的数据处理与数据使用说明](sota/four_model_ensemble_data_usage.md)

## 分数对比

| 路线 | phase1 MAPE | 泄露/选择状态 | 当前建议 |
| --- | ---: | --- | --- |
| ExtraTrees global | `0.120773` | 无泄露 | 稳定基线 |
| low_volume_block | `0.120175` | 无泄露 | 默认单模型路线 |
| 四模型融合 `validate-ensemble` 默认小时权重 | `0.116167` | 无泄露 | 当前最佳可报告验证路线 |
| 四模型融合 `validate-ensemble --weight-scope global` | `0.118018` | 无泄露 | 上一版 SOTA，可复现对照 |
| 四模型融合 `predict-ensemble` 默认小时权重校准 | `0.111638` | 对 phase2 合法；不可当 phase1 无泄露验证分数 | 用于生成 phase2 融合提交 |
| `--history-blend 0.09` | `0.119564` | 使用 phase1 验证标签调权 | 只作上界参考 |
| 轨迹第五候选 block cap `0.15` | `0.115924` | phase1 探索结果；cap 仍需 train1-only 选择协议确认 | 暂不晋升 |
| 观察窗后验校正 phase1 直选最佳 | `0.114456` | phase1 sweep 选出 | exploratory 上界，暂不晋升 |
| 观察窗后验校正 rolling 支持配置 | 约 `0.11583` | train1 rolling 支持更强 | 下一步最值得整理成正式候选 |
| 神经先验门控融合 | `0.114758` | phase1 探索结果；seed 敏感且暂无 rolling 协议 | 值得继续验证，暂不晋升 |

更完整的路线状态、低优先级路线和下一步建议见：

- [当前已得到方案整理](route_exploration_candidates.md)

## 泄露边界

题目允许预测某日红窗时使用该日更早的绿窗数据，即 `06:00-08:00` 和 `15:00-17:00`。不允许使用被预测窗口及其之后的真实流量。

因此：

- 用 train1 训练，并用 phase1 test 绿窗预测 Oct.18-24，是合法验证流程。
- 用 Oct.18-24 的真实标签调 phase1 验证参数，是验证集泄露。
- 在 phase2 阶段，Oct.18-24 已作为新增训练数据发布；用它校准权重后预测 Oct.25-31 是合法的。

## 当前晋升状态

截至本文档更新时，探索路线尚未迁入 `src/`。正式 `src/` 仍保持四模型小时权重融合为默认 SOTA。后续如果要迁入观察窗后验校正，应先在 `src1` 内固定 train1-only 选择规则，再更新 `docs/sota/` 和正式命令说明。
