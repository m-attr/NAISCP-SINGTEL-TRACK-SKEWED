from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from dashboard.theme import PALETTE, PLOT_BACKGROUND, PLOT_GRID


def _compact_history_label(item: dict[str, Any]) -> str:
    from_months = [str(value) for value in item.get("from_months", [])]
    to_months = [str(value) for value in item.get("to_months", [])]
    all_months = from_months + to_months
    years = {value[:2] for value in all_months if len(value) >= 3}
    if len(years) == 1 and all_months:
        year = next(iter(years))

        def names(values: list[str]) -> str:
            labels = [value[3:] if value.startswith(f"{year}-") else value for value in values]
            return "–".join(labels)

        return f"{year} {names(from_months)} → {names(to_months)}"
    return str(item["label"])


def drift_history_figure(feature: dict[str, Any]) -> go.Figure:
    evidence = feature["evidence"]
    history = evidence["historical_shifts"]
    labels = [_compact_history_label(item) for item in history] + [
        "Recent training → current"
    ]
    values = [float(item["shift"]) for item in history] + [
        float(evidence["current_shift"])
    ]
    colors = [PALETTE["historical_light"]] * len(history) + [PALETTE["current"]]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            hovertemplate="%{x}<br>Change %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=315,
        margin=dict(l=16, r=12, t=12, b=80),
        xaxis_title=None,
        yaxis_title="Distribution change",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=PALETTE["text"], size=12),
    )
    figure.update_xaxes(
        tickangle=-28,
        gridcolor=PALETTE["table_header"],
        linecolor=PALETTE["border"],
    )
    figure.update_yaxes(gridcolor=PLOT_GRID, zeroline=False)
    return figure
