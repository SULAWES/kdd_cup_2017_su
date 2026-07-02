from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src3_explore.explain.common import explain_dir


PHASE_CONCLUSIONS = [
    "五节点 message passing 是负贡献；topology 不如 random 说明边缺少稳定交通语义，且单边 harm / heterophily 审计可解释负迁移来源。",
    "这不否定图思想，只否定当前五节点 label graph；route/intersection/tollgate 的 lead-lag process graph 更接近真实机制，但必须严格区分 legal 与 red-window illegal 信号。",
    "LSTM/Transformer 弱不是单纯输入表示问题；raw sequence tree 强于 sequence NN，engineered tree 强于 engineered NN，说明模型族、小样本方差和归纳偏置更关键。",
    "当前复杂方案强在结构化归纳偏置，而不是单模型规模：时间/周周期、green demand state、low-volume 保护、MAPE-aware 训练、模型多样性和 hour scoped blending 共同起作用。",
    "Oracle gap 大但 winner 难预测，下一步应做 margin-aware、top-k、regret-based、selective soft gate，而不是 row-level hard winner classifier。",
]


def write_explain_summary(output_dir: Path, selected: Sequence[str] | None = None) -> Path:
    out_dir = explain_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card_paths = sorted(path for path in out_dir.glob("*.md") if path.name != "explain_summary.md")
    if selected:
        selected_slugs = {name.removeprefix("explain_") for name in selected}
        filtered = []
        for path in card_paths:
            stem = path.stem.removesuffix("_card")
            if stem in selected_slugs or any(stem.startswith(slug) or slug.startswith(stem) for slug in selected_slugs):
                filtered.append(path)
        if filtered:
            card_paths = filtered

    parts = [
        "# src3_explore/explain 阶段性总结",
        "",
        "本文件由 `python -m src3_explore explain_*` 入口汇总生成。所有实验均为解释型诊断，不作为 phase1 反复调参后的正式 SOTA 声明。",
        "",
        "## 阶段性结论",
        "",
    ]
    parts.extend(f"{idx}. {item}" for idx, item in enumerate(PHASE_CONCLUSIONS, start=1))
    parts.extend(["", "## Experiment cards", ""])
    for path in card_paths:
        parts.append(f"<!-- source: {path.name} -->")
        parts.append(path.read_text(encoding="utf-8").strip())
        parts.append("")
    summary_path = out_dir / "explain_summary.md"
    summary_path.write_text("\n".join(parts), encoding="utf-8")
    return summary_path
