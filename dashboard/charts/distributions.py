from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from dashboard.theme import PALETTE, PLOT_BACKGROUND, PLOT_GRID


def _shares(counts: list[int]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def distribution_figure(feature: dict[str, Any]) -> go.Figure:
    distribution = feature["distribution"]
    figure = go.Figure()
    if distribution["kind"] == "numeric":
        edges = np.asarray(distribution["bin_edges"], dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=_shares(distribution["recent_training_counts"]),
                name="Recent training",
                mode="lines",
                line=dict(color=PALETTE["historical"], width=2.2, shape="hv"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=_shares(distribution["test_counts"]),
                name="Current",
                mode="lines",
                line=dict(color=PALETTE["current"], width=2.4, shape="hv"),
            )
        )
        repaired = distribution.get("repaired_test_counts")
        if repaired is not None:
            figure.add_trace(
                go.Scatter(
                    x=centers,
                    y=_shares(repaired),
                    name="Repaired",
                    mode="lines",
                    line=dict(
                        color=PALETTE["repair"],
                        width=2.4,
                        dash="dash",
                        shape="hv",
                    ),
                )
            )
        x_title = str(distribution.get("representation", "Value"))
    else:
        categories = distribution["categories"]
        figure.add_trace(
            go.Bar(
                x=categories,
                y=distribution["recent_training_shares"],
                name="Recent training",
                marker_color=PALETTE["historical"],
            )
        )
        figure.add_trace(
            go.Bar(
                x=categories,
                y=distribution["test_shares"],
                name="Current",
                marker_color=PALETTE["current"],
            )
        )
        repaired = distribution.get("repaired_test_shares")
        if repaired is not None:
            figure.add_trace(
                go.Bar(
                    x=categories,
                    y=repaired,
                    name="Repaired",
                    marker_color=PALETTE["repair"],
                )
            )
        x_title = "Category"

    figure.update_layout(
        height=315,
        margin=dict(l=16, r=12, t=12, b=56),
        barmode="group",
        legend=dict(
            orientation="h",
            traceorder="normal",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
        xaxis_title=x_title,
        yaxis_title="Share of observations",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=PALETTE["text"], size=12),
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor=PALETTE["table_header"], linecolor=PALETTE["border"])
    figure.update_yaxes(gridcolor=PLOT_GRID, tickformat=".0%", zeroline=False)
    return figure
