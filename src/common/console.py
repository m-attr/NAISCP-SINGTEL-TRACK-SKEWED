from __future__ import annotations

from typing import Any


def format_phase_box(content: str, min_inner_width: int = 28) -> str:
    text = str(content).strip()
    inner_width = max(int(min_inner_width), len(text) + 2)
    border = "+" + ("-" * inner_width) + "+"
    line = "|" + text.center(inner_width) + "|"
    return "\n".join([border, line, border])


def format_ascii_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not headers:
        return ""
    safe_headers = [str(value) for value in headers]
    safe_rows = [["" if value is None else str(value) for value in row] for row in rows]
    widths = [len(value) for value in safe_headers]
    for row in safe_rows:
        for index, value in enumerate(row[: len(widths)]):
            widths[index] = max(widths[index], len(value))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(values: list[str]) -> str:
        cells = []
        for index, width in enumerate(widths):
            value = values[index] if index < len(values) else ""
            cells.append(" " + value.ljust(width) + " ")
        return "|" + "|".join(cells) + "|"

    lines = [border, render(safe_headers), border]
    lines.extend(render(row) for row in safe_rows)
    lines.append(border)
    return "\n".join(lines)
