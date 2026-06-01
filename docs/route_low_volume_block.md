# 路线：recent-low-volume block 结构切换

## 核心思路

`low_volume_block` 不是调单个模型参数，而是一个结构切换：

1. 同时训练 `global` 模型和 `block` 模型。
2. 检查每个收费站/方向组合最近 7 天是否进入低位。
3. 若某 combo 最近 7 天均值同时满足：
   - 低于最近 7 天整体均值的 `60%`
   - 低于该 combo 全历史均值的 `60%`
4. 则该 combo 使用 `block` 模型预测；其他 combo 仍使用 `global` 模型。

phase1 和 phase2 训练数据下，该规则都只选择 `1_0`。

## 成绩

命令：

```sh
python run_task2.py validate
```

等价于：

```sh
python run_task2.py --group low_volume_block validate
```

phase1 MAPE：约 `0.120175`。

对比：

- `global`：约 `0.120773`
- `low_volume_block`：约 `0.120175`

## 为什么有效

Oct.11-17 训练期最后一周中，`1_0` 相对自身历史和最近整体流量都明显降低。预测 Oct.18-24 时，如果继续用 global 模型，会受到其他 combo 的正常水平牵引；切到早晚块模型后，`1_0` 的局部分布拟合更好。

## 泄露检查

触发条件只看训练数据的最近 7 天标签：

- phase1 验证时，只看 Sep.19-Oct.17。
- phase2 预测时，可以看 Sep.19-Oct.24，因为 Oct.18-24 已作为新增训练数据发布。

该路线没有使用待预测红窗的真实标签。

## 风险

规则当前依赖最近 7 天 regime 能延续到下一周。如果某个 combo 的低位只是短期异常，切换可能变差。滚动折中，在低位 regime 出现前该规则不会触发，因此不伤分，但历史可验证触发样本较少。
