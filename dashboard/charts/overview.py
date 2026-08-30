from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import (
    ACTION_COLORS,
    ACTION_ORDER,
    PALETTE,
    PLOT_BACKGROUND,
    PLOT_GRID,
)

SEVERITY_LIMIT = 10


def action_distribution_figure(frame: pd.DataFrame) -> go.Figure:
    counts = frame["Action"].value_counts()
    figure = go.Figure()
    for action in ACTION_ORDER:
        value = int(counts.get(action, 0))
        figure.add_trace(
            go.Bar(
                x=[value],
                y=["Features"],
                name=action,
                orientation="h",
                marker_color=ACTION_COLORS[action],
                marker_line_width=0,
                text=[str(value) if value else ""],
                textposition="inside",
                insidetextfont=dict(color=PALETTE["background"], size=12),
                hovertemplate=f"{action}: %{{x:,}} features<extra></extra>",
            )
        )
    figure.update_layout(
        height=104,
        margin=dict(l=8, r=8, t=32, b=8),
        barmode="stack",
        bargap=0.38,
        legend=dict(
            orientation="h",
            traceorder="normal",
            yanchor="bottom",
            y=1.0,
            x=0,
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=PALETTE["text"], size=12),
        showlegend=True,
    )
    return figure


def ranked_severity_rows(
    frame: pd.DataFrame,
    limit: int = SEVERITY_LIMIT,
) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("Severity ranking limit must be positive.")
    return (
        frame.loc[:, ["Feature", "Severity", "Action"]]
        .sort_values(["Severity", "Feature"], ascending=[False, True], kind="stable")
        .head(limit)
        .reset_index(drop=True)
    )


def severity_ranking_figure(
    frame: pd.DataFrame,
    limit: int = SEVERITY_LIMIT,
) -> go.Figure:
    ranked = ranked_severity_rows(frame, limit=limit).iloc[::-1]
    values = ranked["Severity"].astype(float).tolist()
    colors = [
        ACTION_COLORS.get(str(action), PALETTE["keep"])
        for action in ranked["Action"]
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=ranked["Feature"].astype(str).tolist(),
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f}×" for value in values],
            textposition="outside",
            cliponaxis=False,
            customdata=ranked["Action"].astype(str).tolist(),
            hovertemplate=(
                "%{y}<br>Severity %{x:.3f}×<br>Action %{customdata}<extra></extra>"
            ),
        )
    )
    upper = max(values, default=1.0)
    figure.update_layout(
        height=max(260, min(390, 95 + 27 * len(ranked))),
        margin=dict(l=12, r=72, t=8, b=32),
        xaxis=dict(title="Severity", range=[0, upper * 1.18]),
        yaxis=dict(title=None, automargin=True),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=PALETTE["text"], size=12),
        showlegend=False,
    )
    figure.update_xaxes(gridcolor=PLOT_GRID, zeroline=False)
    return figure
