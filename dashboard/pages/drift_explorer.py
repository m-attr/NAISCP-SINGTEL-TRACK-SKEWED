from __future__ import annotations

import hashlib
import math
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.charts.distributions import distribution_figure
from dashboard.charts.evidence import drift_history_figure
from dashboard.components.ui import (
    action_badge,
    chart_caption,
    detail_rows,
    detail_heading,
    direction_indicator,
    metric_card,
    page_header,
)
from dashboard.data.artifacts import feature_by_name, feature_frame
from dashboard.theme import PALETTE, SEVERITY_HELP


TABLE_COLUMNS = [
    "Feature",
    "Status",
    "Current",
    "Historical",
    "Severity",
    "Action",
]
FILTER_COLUMN_WIDTHS = [2.8, 1.2, 1.2, 1.5, 1.4]
LEGEND_TEXT = (
    "<b>How to read this:</b> Current compares recent training with current data. "
    "Historical is the feature's previous movement. Severity compares the current "
    "drift with the feature's historical movement using a stabilised ratio. Higher "
    "values indicate stronger evidence of unusually large drift. Action is the "
    "mitigation decision applied for this run."
)


def _apply_filters(
    frame: pd.DataFrame,
    search: str,
    status: str,
    feature_type: str,
    sort_label: str,
    descending: bool,
) -> pd.DataFrame:
    filtered = frame.copy()
    if search:
        filtered = filtered[
            filtered["Feature"].str.contains(search, case=False, regex=False)
        ]
    if status == "Changed":
        filtered = filtered[filtered["Status"] == "Changed"]
    elif status == "Repaired":
        filtered = filtered[filtered["Action"] == "Repair"]
    elif status == "Dropped":
        filtered = filtered[filtered["Action"] == "Drop"]
    elif status == "Kept":
        filtered = filtered[filtered["Action"] == "Keep"]
    if feature_type != "All types":
        filtered = filtered[filtered["Type"] == feature_type]
    return filtered.sort_values(
        sort_label,
        ascending=not descending,
        kind="stable",
    ).reset_index(drop=True)


def _filter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    with st.container(key="explorer-filters"):
        search_col, status_col, type_col, sort_col, direction_col = st.columns(
            FILTER_COLUMN_WIDTHS,
            gap="small",
        )
        with search_col:
            search = st.text_input("Search", placeholder="Find a feature…")
        with status_col:
            status = st.selectbox(
                "Status",
                ["All", "Changed", "Repaired", "Dropped", "Kept"],
            )
        with type_col:
            feature_type = st.selectbox(
                "Type", ["All types", "Numeric", "Categorical"]
            )
        with sort_col:
            sort_label = st.selectbox(
                "Sort by",
                ["Severity", "Current", "Historical", "Feature", "Action"],
            )
        with direction_col:
            descending = (
                st.selectbox("Direction", ["Descending", "Ascending"])
                == "Descending"
            )
    return _apply_filters(
        frame,
        search=search,
        status=status,
        feature_type=feature_type,
        sort_label=sort_label,
        descending=descending,
    )


def _selected_rows(event: Any) -> list[int]:
    selection = getattr(event, "selection", {})
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows", [])
    return list(rows or [])


def _feature_metrics(feature: dict[str, Any]) -> None:
    evidence = feature["evidence"]
    with st.container(key="explorer-detail-metrics"):
        columns = st.columns(3)
        cards = [
            (
                "Current",
                f'{float(evidence["current_shift"]):.3f}',
                "Current distribution change",
            ),
            (
                "Historical",
                f'{float(evidence["historical_max_shift"]):.3f}',
                "Largest previous movement",
            ),
            (
                "Severity",
                f'{float(evidence["novelty_ratio"]):.2f}×',
                "Compared with its history",
            ),
        ]
        accents = ("blue", "historical", "blue")
        for column, card, accent in zip(columns, cards, accents):
            with column:
                metric_card(*card, accent=accent)


def _has_direction_evidence(feature: dict[str, Any]) -> bool:
    decision = feature["decision"]
    evidence = feature["evidence"]
    train_value = float(evidence.get("orientation_train", 0.0))
    test_value = float(evidence.get("orientation_test", 0.0))
    return (
        decision.get("display_action") == "REPAIR"
        and decision.get("action") == "REVERSE_PERCENTILE"
        and int(evidence.get("orientation_anchor_count", 0)) > 0
        and math.isfinite(train_value)
        and math.isfinite(test_value)
        and train_value * test_value < 0.0
    )


def _severity_scale(frame: pd.DataFrame) -> float:
    maximum = float(frame["Severity"].max()) if not frame.empty else 0.0
    return maximum if math.isfinite(maximum) and maximum > 0.0 else 1.0


def _resolved_selection(frame: pd.DataFrame, requested: str | None) -> str | None:
    if frame.empty:
        return None
    names = frame["Feature"].astype(str).tolist()
    return requested if requested in names else names[0]


def _mitigation_rows(feature: dict[str, Any]) -> list[tuple[str, Any]]:
    decision = feature["decision"]
    action = decision["display_action"]
    rows: list[tuple[str, Any]] = [
        ("Action", action.title()),
        ("Method", decision["technical_method"]),
        ("Reason", decision["reason"]),
    ]
    parameters = decision.get("repair_parameters") or {}
    if parameters.get("estimated_scale_factor") is not None:
        rows.append(
            ("Estimated scale factor", f'{float(parameters["estimated_scale_factor"]):.6g}')
        )
    if parameters.get("estimated_offset") is not None:
        rows.append(
            ("Estimated offset", f'{float(parameters["estimated_offset"]):.6g}')
        )
    if int(parameters.get("mapped_category_count") or 0) > 0:
        rows.append(("Mapped categories", int(parameters["mapped_category_count"])))
        mapping = parameters.get("mapping_summary") or []
        rows.append(
            (
                "Mapping summary",
                ", ".join(
                    f'{item["current"]} → {item["reference"]}'
                    for item in mapping
                ),
            )
        )
    if decision.get("action") == "REVERSE_PERCENTILE":
        rows.append(
            ("Stable references", int(feature["evidence"]["orientation_anchor_count"]))
        )
    return rows


def _render_feature_detail(feature: dict[str, Any]) -> None:
    decision = feature["decision"]
    evidence = feature["evidence"]
    st.markdown(
        '<div class="dd-feature-heading">'
        f'<div class="dd-feature-title">{escape(str(feature["feature"]))}</div>'
        f'{action_badge(decision["display_action"])}'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f'{str(feature["type"]).title()} · Detection method: {feature["detection_method"]}'
    )
    _feature_metrics(feature)

    distribution_col, history_col = st.columns(2, gap="large")
    with distribution_col:
        detail_heading("Distribution comparison")
        st.plotly_chart(
            distribution_figure(feature),
            width="stretch",
            config={"displayModeBar": False},
            key=f'detail-distribution-{feature["feature"]}',
        )
        distribution = feature["distribution"]
        chart_caption(
            f'{int(distribution["recent_training_observations"]):,} recent-training and '
            f'{int(distribution["test_observations"]):,} current observations.'
        )
    with history_col:
        detail_heading("Historical drift")
        st.plotly_chart(
            drift_history_figure(feature),
            width="stretch",
            config={"displayModeBar": False},
            key=f'detail-history-{feature["feature"]}',
        )
        chart_caption(
            "The current comparison is highlighted against chronological training history."
        )

    if _has_direction_evidence(feature):
        detail_heading("Direction evidence", separated=True)
        direction_indicator(feature)

    detail_heading("Mitigation", separated=True)
    detail_rows(_mitigation_rows(feature))
    if decision["display_action"] != "KEEP":
        st.caption(
            f'Predictive strength {float(evidence["predictive_strength"]):.3f} · '
            f'Missingness change {float(evidence["missing_shift"]):.3f} · '
            f'Support retained {float(evidence["support_retention_ratio"]):.1%}'
        )


def render(payload: dict[str, Any]) -> None:
    page_header(
        "Drift Explorer",
        "Explore feature-level changes, distributions, and mitigation decisions.",
    )
    full_frame = feature_frame(payload)
    severity_max = _severity_scale(full_frame)
    frame = _filter_frame(full_frame)
    if frame.empty:
        st.session_state.pop("selected_feature_name", None)
        st.caption("0 features")
        st.markdown(
            '<div class="dd-placeholder">No features match the current search and filters.</div>',
            unsafe_allow_html=True,
        )
        return

    table = frame[TABLE_COLUMNS]
    table_signature = hashlib.sha256(
        "\x1f".join(table["Feature"].astype(str)).encode("utf-8")
    ).hexdigest()[:12]
    if st.session_state.get("feature_table_signature") != table_signature:
        st.session_state["feature_table_signature"] = table_signature
        st.session_state["selected_feature_name"] = str(table.iloc[0]["Feature"])
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=330,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width=270),
            "Status": st.column_config.TextColumn(
                "Status",
                help="Whether the run found evidence requiring investigation.",
                width=80,
            ),
            "Current": st.column_config.NumberColumn(
                "Current",
                help="Difference between the recent training reference and current data.",
                format="%.3f",
                width=80,
            ),
            "Historical": st.column_config.NumberColumn(
                "Historical",
                help="Previous movement used to establish this feature's historical behaviour.",
                format="%.3f",
                width=86,
            ),
            "Severity": st.column_config.ProgressColumn(
                "Severity",
                help=SEVERITY_HELP,
                format="%.2f×",
                min_value=0.0,
                max_value=severity_max,
                step=0.01,
                color=PALETTE["primary"],
                width=145,
            ),
            "Action": st.column_config.TextColumn(
                "Action",
                help="Mitigation decision applied before final prediction.",
                width=75,
            ),
        },
        key=f"feature-evidence-table-{table_signature}",
    )
    st.markdown(
        f'<div class="dd-legend"><b>{len(frame)} feature'
        f'{"s" if len(frame) != 1 else ""}.</b> {LEGEND_TEXT}</div>',
        unsafe_allow_html=True,
    )

    rows = _selected_rows(event)
    if rows:
        selected_name = str(table.iloc[int(rows[0])]["Feature"])
        st.session_state["selected_feature_name"] = selected_name
    selected_name = _resolved_selection(
        table,
        st.session_state.get("selected_feature_name"),
    )
    st.session_state["selected_feature_name"] = selected_name
    _render_feature_detail(feature_by_name(payload, str(selected_name)))
