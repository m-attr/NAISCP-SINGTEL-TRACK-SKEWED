from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.charts.distributions import distribution_figure
from dashboard.charts.evidence import _compact_history_label, drift_history_figure
from dashboard.charts.overview import (
    ACTION_COLORS,
    ACTION_ORDER,
    SEVERITY_LIMIT,
    action_distribution_figure,
    ranked_severity_rows,
    severity_ranking_figure,
)
from dashboard.charts.runtime import runtime_figure
from dashboard.data.artifacts import feature_frame, load_dashboard_run
from dashboard.pages.drift_explorer import (
    FILTER_COLUMN_WIDTHS,
    LEGEND_TEXT,
    TABLE_COLUMNS,
    _apply_filters,
    _has_direction_evidence,
    _mitigation_rows,
    _resolved_selection,
    _severity_scale,
)
from dashboard.pages.overview import _affected_table_height
from dashboard.pages.performance_runtime import _processing_frame
from dashboard.theme import PALETTE, SEVERITY_HELP, SPACING


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "dashboard_artifacts" / "dashboard_run.json"


@pytest.fixture(scope="module")
def dashboard_run() -> dict:
    if not ARTIFACT_PATH.is_file():
        pytest.skip("Real dashboard artifact has not been generated.")
    return load_dashboard_run(ARTIFACT_PATH)


def _feature_for_action(payload: dict, action: str) -> dict:
    return next(
        feature
        for feature in payload["features"]
        if feature["decision"]["display_action"] == action
    )


def _synthetic_feature_frame(total: int, affected: int) -> pd.DataFrame:
    if not 0 <= affected <= total:
        raise ValueError("Affected feature count must fit within the total.")
    rows = []
    for index in range(total):
        if index < affected:
            action = "Repair" if index % 2 == 0 else "Drop"
            status = "Changed"
        else:
            action = "Keep"
            status = "Stable"
        rows.append(
            {
                "Feature": f"AnonymousSignal{index:03d}",
                "Type": "Numeric" if index % 2 == 0 else "Categorical",
                "Status": status,
                "Current": (total - index) / max(total, 1),
                "Historical": (index + 1) / max(total * 10, 1),
                "Severity": float(total - index) / 7.0,
                "Action": action,
            }
        )
    return pd.DataFrame(rows)


def test_explorer_primary_table_and_single_row_filter_contract(dashboard_run: dict) -> None:
    frame = feature_frame(dashboard_run)
    assert TABLE_COLUMNS == [
        "Feature",
        "Status",
        "Current",
        "Historical",
        "Severity",
        "Action",
    ]
    assert all(column in frame for column in TABLE_COLUMNS)
    assert "Historical normal" not in frame
    assert "Unusualness" not in frame
    assert FILTER_COLUMN_WIDTHS == [2.8, 1.2, 1.2, 1.5, 1.4]
    assert "How to read this" in LEGEND_TEXT


def test_explorer_search_filters_and_sorting_are_deterministic(dashboard_run: dict) -> None:
    frame = feature_frame(dashboard_run)
    repaired = _feature_for_action(dashboard_run, "REPAIR")
    repaired_name = repaired["feature"]

    searched = _apply_filters(
        frame,
        search=repaired_name.lower(),
        status="All",
        feature_type="All types",
        sort_label="Feature",
        descending=False,
    )
    assert searched["Feature"].tolist() == [repaired_name]

    changed = _apply_filters(
        frame,
        search="",
        status="Changed",
        feature_type="All types",
        sort_label="Severity",
        descending=True,
    )
    assert set(changed["Action"]) == {"Repair", "Drop"}
    assert changed["Severity"].is_monotonic_decreasing

    numeric = _apply_filters(
        frame,
        search="",
        status="All",
        feature_type="Numeric",
        sort_label="Current",
        descending=False,
    )
    assert set(numeric["Type"]) == {"Numeric"}
    assert numeric["Current"].is_monotonic_increasing


def test_explorer_severity_scale_and_default_selection_use_the_full_run(
    dashboard_run: dict,
) -> None:
    frame = feature_frame(dashboard_run)
    sorted_frame = _apply_filters(
        frame,
        search="",
        status="All",
        feature_type="All types",
        sort_label="Severity",
        descending=True,
    )
    changed_only = sorted_frame[sorted_frame["Status"] == "Changed"].reset_index(drop=True)

    assert _severity_scale(frame) == pytest.approx(float(frame["Severity"].max()))
    assert _severity_scale(changed_only) <= _severity_scale(frame)
    assert _resolved_selection(sorted_frame, None) == str(sorted_frame.iloc[0]["Feature"])
    assert _resolved_selection(sorted_frame, str(sorted_frame.iloc[-1]["Feature"])) == str(
        sorted_frame.iloc[-1]["Feature"]
    )
    assert _resolved_selection(changed_only, "not-in-filter") == str(
        changed_only.iloc[0]["Feature"]
    )
    assert _resolved_selection(sorted_frame.iloc[0:0], None) is None


def test_overview_uses_real_counts_and_bounded_top_n_severity(dashboard_run: dict) -> None:
    frame = feature_frame(dashboard_run)
    action_chart = action_distribution_figure(frame)
    chart_counts = {
        trace.name: int(trace.x[0])
        for trace in action_chart.data
    }
    expected_counts = frame["Action"].value_counts().to_dict()
    assert tuple(trace.name for trace in action_chart.data) == ACTION_ORDER
    assert chart_counts == {
        action: int(expected_counts.get(action, 0))
        for action in ACTION_ORDER
    }

    ranked = ranked_severity_rows(frame)
    assert len(ranked) == min(SEVERITY_LIMIT, len(frame))
    assert ranked["Severity"].is_monotonic_decreasing
    severity_chart = severity_ranking_figure(frame)
    assert list(reversed(severity_chart.data[0].x)) == pytest.approx(
        ranked["Severity"].tolist()
    )
    assert int(severity_chart.layout.height) <= 390


@pytest.mark.parametrize("affected", [10, 30])
def test_overview_affected_table_stays_bounded(affected: int) -> None:
    frame = _synthetic_feature_frame(total=max(affected + 5, 40), affected=affected)
    affected_frame = frame[frame["Action"] != "Keep"]
    assert len(affected_frame) == affected
    assert _affected_table_height(len(affected_frame)) <= 360


def test_dashboard_charts_scale_to_more_than_one_hundred_features() -> None:
    frame = _synthetic_feature_frame(total=128, affected=30)
    ranked = ranked_severity_rows(frame)
    action_chart = action_distribution_figure(frame)
    severity_chart = severity_ranking_figure(frame)

    assert len(frame) > 100
    assert len(ranked) == SEVERITY_LIMIT
    assert sum(int(trace.x[0]) for trace in action_chart.data) == len(frame)
    assert len(severity_chart.data[0].x) == SEVERITY_LIMIT
    assert int(severity_chart.layout.height) <= 390


def test_feature_investigation_uses_real_distribution_and_history(dashboard_run: dict) -> None:
    repaired = _feature_for_action(dashboard_run, "REPAIR")
    dropped = _feature_for_action(dashboard_run, "DROP")
    kept = _feature_for_action(dashboard_run, "KEEP")

    repaired_distribution = distribution_figure(repaired)
    assert [trace.name for trace in repaired_distribution.data] == [
        "Recent training",
        "Current",
        "Repaired",
    ]
    for feature in (dropped, kept):
        figure = distribution_figure(feature)
        assert [trace.name for trace in figure.data] == ["Recent training", "Current"]

    for feature in (repaired, dropped, kept):
        history = drift_history_figure(feature)
        expected_labels = [
            _compact_history_label(item)
            for item in feature["evidence"]["historical_shifts"]
        ] + ["Recent training → current"]
        expected_values = [
            float(item["shift"])
            for item in feature["evidence"]["historical_shifts"]
        ] + [float(feature["evidence"]["current_shift"])]
        assert list(history.data[0].x) == expected_labels
        np.testing.assert_allclose(list(history.data[0].y), expected_values)


def test_mitigation_summary_is_action_based_and_conditional(dashboard_run: dict) -> None:
    repaired = _feature_for_action(dashboard_run, "REPAIR")
    dropped = _feature_for_action(dashboard_run, "DROP")
    kept = _feature_for_action(dashboard_run, "KEEP")

    repaired_rows = dict(_mitigation_rows(repaired))
    assert repaired_rows["Action"] == "Repair"
    assert repaired_rows["Stable references"] == repaired["evidence"][
        "orientation_anchor_count"
    ]
    assert "Stable references" not in dict(_mitigation_rows(dropped))
    kept_rows = dict(_mitigation_rows(kept))
    assert kept_rows["Action"] == "Keep"
    assert kept_rows["Method"] == kept["decision"]["technical_method"]
    assert kept_rows["Reason"] == kept["decision"]["reason"]
    assert _has_direction_evidence(repaired) is True
    assert _has_direction_evidence(dropped) is False
    assert _has_direction_evidence(kept) is False

    unrelated_repair = deepcopy(repaired)
    unrelated_repair["decision"]["action"] = "FUTURE_NON_DIRECTIONAL_REPAIR"
    assert _has_direction_evidence(unrelated_repair) is False


def test_mitigation_details_render_all_supported_methods_generically(
    dashboard_run: dict,
) -> None:
    template = deepcopy(_feature_for_action(dashboard_run, "KEEP"))
    scenarios = [
        ("KEEP", "KEEP", "No mitigation applied", {}),
        (
            "KEEP_DRIFTED",
            "KEEP",
            "Preserve same-semantic population drift",
            {},
        ),
        ("KEEP_SYSTEMIC_DRIFT", "KEEP", "Preserve systemic drift", {}),
        (
            "REALIGN_SCALE",
            "REPAIR",
            "Scale realignment",
            {"estimated_scale_factor": 100.0},
        ),
        (
            "REALIGN_OFFSET",
            "REPAIR",
            "Offset realignment",
            {"estimated_offset": 100.0},
        ),
        (
            "REMAP_CATEGORIES",
            "REPAIR",
            "Category remapping",
            {
                "mapped_category_count": 2,
                "mapping_summary": [
                    {"current": "new_1", "reference": "old_1"},
                    {"current": "new_2", "reference": "old_2"},
                ],
            },
        ),
        (
            "REVERSE_PERCENTILE",
            "REPAIR",
            "Reverse percentile representation",
            {},
        ),
        (
            "DROP_NOVEL_UNRELIABLE",
            "DROP",
            "Exclude unreliable feature",
            {},
        ),
    ]
    for action, display, method, parameters in scenarios:
        feature = deepcopy(template)
        feature["decision"] = {
            "action": action,
            "display_action": display,
            "technical_method": method,
            "reason": "Anonymous artifact-backed reason.",
            "repair_parameters": parameters,
        }
        rows = dict(_mitigation_rows(feature))
        assert rows["Action"] == display.title()
        assert rows["Method"] == method
        assert rows["Reason"] == "Anonymous artifact-backed reason."
        if action == "REALIGN_SCALE":
            assert rows["Estimated scale factor"] == "100"
        if action == "REALIGN_OFFSET":
            assert rows["Estimated offset"] == "100"
        if action == "REMAP_CATEGORIES":
            assert rows["Mapped categories"] == 2
            assert "new_1 → old_1" in rows["Mapping summary"]


def test_combined_performance_runtime_page_uses_current_run_evidence(
    dashboard_run: dict,
) -> None:
    metrics = dashboard_run["run"]["metrics"]
    processing = _processing_frame(metrics)
    assert processing.columns.tolist() == [
        "Stage",
        "Rows scanned",
        "Rows retained / written",
    ]
    assert processing["Rows scanned"].tolist() == [
        metrics["train_rows_scanned"],
        metrics["train_rows_scanned"],
        metrics["test_rows_scanned"],
        metrics["test_rows_scanned"],
    ]
    assert dashboard_run["run"]["model_contract"]["fit_count"] == 1


def test_semantic_palette_and_spacing_tokens_are_centralised() -> None:
    assert PALETTE == {
        "primary": "#2563EB",
        "current": "#2563EB",
        "historical": "#94A3B8",
        "historical_light": "#B2BCC7",
        "keep": "#526579",
        "repair": "#16A36E",
        "drop": "#DC3F45",
        "text": "#202731",
        "muted": "#64748B",
        "border": "#D8E0E8",
        "border_strong": "#CBD5E1",
        "table_header": "#F1F5F9",
        "selected": "#EAF2FF",
        "background": "#FFFFFF",
        "surface_subtle": "#F8FAFC",
    }
    assert SPACING == {
        "xs": "8px",
        "sm": "12px",
        "md": "16px",
        "lg": "24px",
        "xl": "32px",
    }
    ui_source = (REPO_ROOT / "dashboard" / "components" / "ui.py").read_text(
        encoding="utf-8"
    )
    assert "--space-xs" in ui_source
    assert "--space-xl" in ui_source
    assert "var(--dd-selected)" in ui_source
    assert "var(--dd-table-header)" in ui_source
    dashboard_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "dashboard").rglob("*.py"))
    ).lower()
    for retired_colour in (
        "#567a9d",
        "#78838f",
        "#4d806d",
        "#a95151",
        "#7894ad",
    ):
        assert retired_colour not in dashboard_source


def test_action_palette_order_and_zero_drop_render_cleanly() -> None:
    frame = _synthetic_feature_frame(total=5, affected=1)
    action_chart = action_distribution_figure(frame)
    assert tuple(trace.name for trace in action_chart.data) == ACTION_ORDER
    assert ACTION_ORDER == ("Keep", "Repair", "Drop")
    assert ACTION_COLORS == {
        "Keep": PALETTE["keep"],
        "Repair": PALETTE["repair"],
        "Drop": PALETTE["drop"],
    }
    assert [trace.marker.color for trace in action_chart.data] == [
        PALETTE["keep"],
        PALETTE["repair"],
        PALETTE["drop"],
    ]
    assert [int(trace.x[0]) for trace in action_chart.data] == [4, 1, 0]
    assert action_chart.data[-1].text[0] == ""
    assert action_chart.layout.legend.traceorder == "normal"

    severity_chart = severity_ranking_figure(frame)
    actions = list(severity_chart.data[0].customdata)
    colors = list(severity_chart.data[0].marker.color)
    assert colors == [ACTION_COLORS[action] for action in actions]


def test_chart_palette_is_consistent_for_reference_current_and_repaired(
    dashboard_run: dict,
) -> None:
    repaired = _feature_for_action(dashboard_run, "REPAIR")
    distribution = distribution_figure(repaired)
    assert [trace.name for trace in distribution.data] == [
        "Recent training",
        "Current",
        "Repaired",
    ]
    assert [trace.line.color for trace in distribution.data] == [
        PALETTE["historical"],
        PALETTE["current"],
        PALETTE["repair"],
    ]
    assert [trace.line.shape for trace in distribution.data] == ["hv", "hv", "hv"]

    categorical = next(
        feature
        for feature in dashboard_run["features"]
        if feature["distribution"]["kind"] == "categorical"
        and feature["distribution"].get("repaired_test_shares") is None
    )
    categorical_chart = distribution_figure(categorical)
    assert [trace.marker.color for trace in categorical_chart.data] == [
        PALETTE["historical"],
        PALETTE["current"],
    ]

    history = drift_history_figure(repaired)
    assert list(history.data[0].marker.color)[-1] == PALETTE["current"]
    assert set(list(history.data[0].marker.color)[:-1]) == {
        PALETTE["historical_light"]
    }


def test_runtime_and_severity_language_follow_the_visual_contract(
    dashboard_run: dict,
) -> None:
    runtime = runtime_figure(dashboard_run["run"]["metrics"])
    assert runtime.data[0].marker.color == PALETTE["primary"]
    assert "stabilised ratio" in SEVERITY_HELP
    assert "Higher values" in SEVERITY_HELP
    assert "stabilised ratio" in LEGEND_TEXT

    config = (REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'primaryColor = "#2563EB"' in config
    assert 'secondaryBackgroundColor = "#F1F5F9"' in config
    assert 'textColor = "#202731"' in config


def test_navigation_has_exactly_three_consolidated_pages() -> None:
    source = (REPO_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert source.count("st.Page(") == 3
    assert 'title="Overview"' in source
    assert 'title="Drift Explorer"' in source
    assert 'title="Performance & Runtime"' in source
    assert 'title="Performance"' not in source
    assert 'title="Runtime & Scalability"' not in source
    assert source.count("st.page_link(") == 3


def test_dashboard_source_has_no_legacy_presentation_or_feature_specific_logic(
    dashboard_run: dict,
) -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "dashboard").rglob("*.py"))
    )
    removed = [
        "REAL RUN MODE",
        "CURRENT VALIDATED RUN",
        "FEATURE-LEVEL EVIDENCE",
        "ONE-MODEL OUTCOME",
        "BOUNDED-MEMORY EXECUTION",
        "Safety boundary",
        "Historical development runs",
        "Future scale benchmarks",
        "@st.dialog",
        "feature_summary_card",
        "surprise_figure",
        "direction_figure",
        "Unusualness",
    ]
    for text in removed:
        assert text not in source

    affected_names = {
        feature["feature"]
        for feature in dashboard_run["features"]
        if feature["decision"]["display_action"] != "KEEP"
    }
    assert not [name for name in affected_names if name in source]
