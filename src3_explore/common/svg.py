from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping, Sequence


def write_bar_svg(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    label_field: str,
    value_field: str,
    title: str,
    max_items: int = 20,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(rows)[:max_items]
    width = 760
    row_h = 28
    left = 210
    top = 48
    height = top + row_h * max(1, len(items)) + 28
    values = [abs(float(row.get(value_field, 0.0))) for row in items]
    vmax = max(values) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="Segoe UI, Arial" font-size="18" font-weight="600">{escape(title)}</text>',
    ]
    for idx, row in enumerate(items):
        y = top + idx * row_h
        value = float(row.get(value_field, 0.0))
        label = str(row.get(label_field, ""))
        bar_w = 0.0 if vmax <= 0 else min(500.0, 500.0 * abs(value) / vmax)
        color = "#2563eb" if value >= 0 else "#dc2626"
        parts.append(f'<text x="24" y="{y + 17}" font-family="Segoe UI, Arial" font-size="12">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.2f}" height="16" fill="{color}" opacity="0.85"/>')
        parts.append(
            f'<text x="{left + bar_w + 8:.2f}" y="{y + 17}" font-family="Segoe UI, Arial" font-size="12">'
            f"{value:.4f}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path

