from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from dashboard.theme import PALETTE, PLOT_BACKGROUND, PLOT_GRID


PHASE_LABELS = {
    "bounded_train_test_analysis_scans": "Scan & profile",
    "drift_plan_and_train_preparation": "Detect & create plan",
    "model_training": "Train final model",
    "streamed_test_transform_prediction_and_output": "Predict & write",
    "artifact_finalization": "Finalise artifacts",
}


def runtime_figure(metrics: dict[str, Any]) -> go.Figure:
    timings = metrics.get("timings_seconds", {})
    keys = [key for key in PHASE_LABELS if key in timings]
    labels = [PHASE_LABELS[key] for key in keys]
    values = [float(timings[key]) for key in keys]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=PALETTE["primary"],
            text=[f"{value:.3f}s" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.4f} seconds<extra></extra>",
        )
    )
    figure.update_layout(
        height=340,
        margin=dict(l=24, r=64, t=16, b=32),
        xaxis=dict(
            title="Seconds",
            range=[0, max(values, default=1.0) * 1.22],
        ),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=PALETTE["text"], size=12),
    )
    figure.update_xaxes(gridcolor=PLOT_GRID, zeroline=False)
    return figure
