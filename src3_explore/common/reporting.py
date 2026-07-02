from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ExperimentCard:
    name: str
    hypothesis: str
    data_visibility: str
    prototype: str
    metrics: Mapping[str, object]
    result: str
    insight: str
    next_step: str
    artifacts: Sequence[str] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        metric_lines = "\n".join(f"- `{key}`: {value}" for key, value in self.metrics.items()) or "- 无"
        artifact_lines = "\n".join(f"- `{item}`" for item in self.artifacts) or "- 无"
        return "\n".join(
            [
                f"## {self.name}",
                "",
                f"**假设（Hypothesis）:** {self.hypothesis}",
                "",
                f"**数据可见性（Data visibility）:** {self.data_visibility}",
                "",
                f"**最小实现（Prototype）:** {self.prototype}",
                "",
                f"**预期洞察（Expected insight）:** {self.insight}",
                "",
                "**指标（Metrics）:**",
                metric_lines,
                "",
                f"**结果（Result）:** {self.result}",
                "",
                f"**下一步（Next）:** {self.next_step}",
                "",
                "**产物（Artifacts）:**",
                artifact_lines,
                "",
            ]
        )


def safe_slug(name: str) -> str:
    keep = []
    for char in name.lower():
        if char.isalnum():
            keep.append(char)
        elif char in (" ", "-", "_", "/"):
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "experiment"


def write_card(output_dir: Path, card: ExperimentCard) -> Path:
    card_dir = output_dir / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    path = card_dir / f"{safe_slug(card.name)}.md"
    path.write_text(card.to_markdown(), encoding="utf-8")
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    names.append(key)
                    seen.add(key)
        fieldnames = names
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
