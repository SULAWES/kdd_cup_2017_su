# 路线：history_blend 后处理上界实验

## 核心思路

`history_blend` 是一个简单后处理：把模型预测和 7 天前同一收费站/方向/时段的真实流量做线性融合。

形式：

```text
adjusted = (1 - blend) * model_pred + blend * lag_7
```

该后处理利用交通流量的周周期性。它本身不复杂，但权重选择非常敏感。

## 成绩

命令：

```sh
python run_task2.py --history-blend 0.09 validate
```

phase1 MAPE：约 `0.119564`。

更早版本中还试过：

```sh
python run_task2.py --sample-weight-power 0.22 --history-blend 0.195 --prediction-scale 0.962 validate
```

该组合在旧模型上约 `0.117796`。

## 泄露状态

这条路线当前不能作为默认无泄露方案。

原因是较优的 blend 权重来自 phase1 验证标签调参。训练期滚动周没有支持把该权重固定为默认值；有些折上 blend 会明显变差。

## 适用价值

该路线适合作为上界实验：

- 说明模型误差中存在可由周周期修正的部分。
- 给后续“只用训练折拟合后处理参数”的方法提供方向。
- 不应直接用于报告无泄露 phase1 验证成绩。
