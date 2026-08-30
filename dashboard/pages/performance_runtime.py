from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.charts.runtime import runtime_figure
from dashboard.components.ui import chart_caption, metric_card, page_header, section


def _memory(value: Any) -> str:
    return "Not measured" if value is None else f"{float(value) / (1024 ** 2):.1f} MiB"


def _processing_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Stage": "Training model",
                "Rows scanned": int(metrics["train_rows_scanned"]),
                "Rows retained / written": int(metrics["train_rows_used"]),
            },
            {
                "Stage": "Training drift reference",
                "Rows scanned": int(metrics["train_rows_scanned"]),
                "Rows retained / written": int(metrics["drift_analysis_train_rows"]),
            },
            {
                "Stage": "Current-data reference",
                "Rows scanned": int(metrics["test_rows_scanned"]),
                "Rows retained / written": int(metrics["test_analysis_rows"]),
            },
            {
                "Stage": "Prediction output",
                "Rows scanned": int(metrics["test_rows_scanned"]),
                "Rows retained / written": int(metrics["prediction_rows"]),
            },
        ]
    )


def render(payload: dict[str, Any]) -> None:
    page_header(
        "Performance & Runtime",
        "Model quality and measured operational evidence for this run.",
    )
    run = payload["run"]
    external = run["external_evaluation"]
    score = external.get("public_test_auprc")
    performance_columns = st.columns(3)
    performance_cards = [
        ("Train AU-PRC", f'{float(run["train_auprc"]):.4f}', "Training diagnostic", "blue"),
        (
            "Public Test AU-PRC",
            "Not available" if score is None else f"{float(score):.4f}",
            "External evaluation" if score is not None else "No evaluation artifact",
            "blue",
        ),
        (
            "Model Fits",
            str(run["model_contract"]["fit_count"]),
            "Production run",
            "historical",
        ),
    ]
    for column, card in zip(performance_columns, performance_cards):
        with column:
            metric_card(*card)

    metrics = run["metrics"]
    streaming = metrics["streaming"]
    rows_processed = int(metrics["train_rows_scanned"]) + int(metrics["test_rows_scanned"])
    section("Runtime", "Measured pipeline execution and bounded data processing.")
    runtime_columns = st.columns(4)
    runtime_cards = [
        ("Total runtime", f'{float(metrics["timings_seconds"]["total"]):.2f}s', "Production pipeline", "blue"),
        (
            "Peak memory",
            _memory(metrics.get("peak_process_rss_bytes")),
            "Process RSS",
            "historical",
        ),
        (
            "Largest prediction chunk",
            f'{int(streaming["maximum_prediction_chunk_rows"]):,}',
            "Rows",
            "historical",
        ),
        ("Dataset rows", f"{rows_processed:,}", "Train + current data", "blue"),
    ]
    for column, card in zip(runtime_columns, runtime_cards):
        with column:
            metric_card(*card)

    section("Phase timing")
    st.plotly_chart(
        runtime_figure(metrics),
        width="stretch",
        config={"displayModeBar": False},
        key="runtime-phase-chart",
    )

    section("Data processing")
    st.dataframe(
        _processing_frame(metrics),
        hide_index=True,
        width="stretch",
        height=180,
        column_config={
            "Stage": st.column_config.TextColumn(width="large"),
            "Rows scanned": st.column_config.NumberColumn(format="localized", width="small"),
            "Rows retained / written": st.column_config.NumberColumn(format="localized", width="medium"),
        },
        key="runtime-row-table",
    )
    chart_caption(
        "Data is processed in bounded chunks rather than loading the complete dataset into memory."
    )
