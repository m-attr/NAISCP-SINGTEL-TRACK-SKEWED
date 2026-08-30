from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.charts.overview import (
    action_distribution_figure,
    severity_ranking_figure,
)
from dashboard.components.ui import metric_card, page_header, section
from dashboard.data.artifacts import feature_frame
from dashboard.theme import SEVERITY_HELP


def _affected_table_height(row_count: int) -> int:
    return min(360, 40 + max(1, row_count) * 35)


def render(payload: dict[str, Any]) -> None:
    page_header(
        "Overview",
        "Summary of detected feature drift and mitigation actions for this run.",
    )
    summary = payload["summary"]
    columns = st.columns(4)
    cards = [
        ("Features analysed", str(summary["features_analyzed"]), "", "blue"),
        ("Kept", str(summary["features_kept"]), "No intervention", "keep"),
        ("Repaired", str(summary["features_repaired"]), "Representation adjusted", "repair"),
        ("Dropped", str(summary["features_dropped"]), "Excluded from preparation", "drop"),
    ]
    for column, card in zip(columns, cards):
        with column:
            metric_card(*card)

    frame = feature_frame(payload)
    section("Action distribution")
    st.plotly_chart(
        action_distribution_figure(frame),
        width="stretch",
        config={"displayModeBar": False},
        key="overview-action-distribution",
    )

    section("Features requiring attention", "Features with a repair or drop decision.")
    affected = frame
    affected = affected.loc[
        affected["Action"] != "Keep",
        ["Feature", "Current", "Historical", "Severity", "Action"],
    ].reset_index(drop=True)
    if affected.empty:
        st.caption("No feature required mitigation in this run.")
    else:
        st.dataframe(
            affected,
            hide_index=True,
            width="stretch",
            height=_affected_table_height(len(affected)),
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="large"),
                "Current": st.column_config.NumberColumn(
                    "Current",
                    help="Difference between recent training and current data.",
                    format="%.3f",
                    width="small",
                ),
                "Historical": st.column_config.NumberColumn(
                    "Historical",
                    help="Largest previous movement recorded for this feature.",
                    format="%.3f",
                    width="small",
                ),
                "Severity": st.column_config.NumberColumn(
                    "Severity",
                    help=SEVERITY_HELP,
                    format="%.2f×",
                    width="small",
                ),
                "Action": st.column_config.TextColumn("Action", width="small"),
            },
            key="overview-affected-features",
        )

    section(
        "Drift severity",
        "Highest stabilised current-to-historical Severity ratios for this run (top 10).",
    )
    st.plotly_chart(
        severity_ranking_figure(frame),
        width="stretch",
        config={"displayModeBar": False},
        key="overview-severity-ranking",
    )
