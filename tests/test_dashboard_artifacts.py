from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from common.contracts import parse_competition_month_value
from dashboard.data.artifacts import (
    ArtifactError,
    load_dashboard_run,
    validate_dashboard_run,
)
from tools.build_dashboard_artifacts import _decision_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "dashboard_artifacts" / "dashboard_run.json"


@pytest.fixture(scope="module")
def dashboard_run() -> dict:
    if not ARTIFACT_PATH.is_file():
        pytest.skip("Real dashboard artifact has not been generated.")
    return load_dashboard_run(ARTIFACT_PATH)


def test_dashboard_schema_actions_and_feature_references(dashboard_run: dict) -> None:
    validate_dashboard_run(dashboard_run)
    assert dashboard_run["schema"] == {
        "name": "datadrift.dashboard_run",
        "version": "1.0.0",
    }
    assert dashboard_run["mode"] == "REAL_RUN"
    assert dashboard_run["source"]["bounded_summary_only"] is True
    assert dashboard_run["source"]["test_target_read"] is False

    features = dashboard_run["features"]
    names = [feature["feature"] for feature in features]
    assert len(names) == len(set(names)) == dashboard_run["summary"]["features_analyzed"]
    assert {feature["decision"]["display_action"] for feature in features} == {
        "KEEP",
        "DROP",
        "REPAIR",
    }


def test_dashboard_histories_are_chronological(dashboard_run: dict) -> None:
    for feature in dashboard_run["features"]:
        history = feature["evidence"]["historical_shifts"]
        for item in history:
            from_values = [parse_competition_month_value(value) for value in item["from_months"]]
            to_values = [parse_competition_month_value(value) for value in item["to_months"]]
            assert max(from_values) < min(to_values)
            assert float(item["shift"]) >= 0.0
        if history:
            expected_max = max(float(item["shift"]) for item in history)
            assert float(feature["evidence"]["historical_max_shift"]) == pytest.approx(
                expected_max,
                abs=1e-12,
            )


def test_dashboard_distributions_are_bounded_and_valid(dashboard_run: dict) -> None:
    for feature in dashboard_run["features"]:
        distribution = feature["distribution"]
        if distribution["kind"] == "numeric":
            edges = np.asarray(distribution["bin_edges"], dtype=np.float64)
            assert len(edges) == len(distribution["recent_training_counts"]) + 1
            assert len(edges) == len(distribution["test_counts"]) + 1
            assert np.isfinite(edges).all()
            assert np.all(np.diff(edges) > 0)
            repaired = distribution.get("repaired_test_counts")
            if repaired is not None:
                assert len(repaired) == len(edges) - 1
        else:
            categories = distribution["categories"]
            assert len(categories) <= 25
            assert len(categories) == len(distribution["recent_training_counts"])
            assert len(categories) == len(distribution["test_counts"])
            assert sum(distribution["recent_training_shares"]) == pytest.approx(1.0)
            assert sum(distribution["test_shares"]) == pytest.approx(1.0)


def test_dashboard_contains_no_raw_test_target(dashboard_run: dict) -> None:
    serialized = json.dumps(dashboard_run, sort_keys=True)
    assert "ChurnStatus" not in serialized
    assert '"test_target_read": false' in serialized


def test_optional_external_evaluation_may_be_absent(dashboard_run: dict) -> None:
    copied = json.loads(json.dumps(dashboard_run))
    copied["run"]["external_evaluation"] = {
        "status": "not_available",
        "label": "External labelled evaluation",
        "public_test_auprc": None,
    }
    validate_dashboard_run(copied)


def test_dashboard_schema_accepts_all_generic_multidrift_methods(
    dashboard_run: dict,
) -> None:
    copied = deepcopy(dashboard_run)
    actions = [
        ("KEEP", "KEEP"),
        ("KEEP_DRIFTED", "KEEP"),
        ("KEEP_SYSTEMIC_DRIFT", "KEEP"),
        ("REALIGN_SCALE", "REPAIR"),
        ("REALIGN_OFFSET", "REPAIR"),
        ("REMAP_CATEGORIES", "REPAIR"),
        ("REVERSE_PERCENTILE", "REPAIR"),
        ("DROP_NOVEL_UNRELIABLE", "DROP"),
    ]
    copied["features"] = deepcopy(copied["features"][: len(actions)])
    for feature, (action, display) in zip(copied["features"], actions):
        feature["decision"].update(
            {
                "action": action,
                "display_action": display,
                "technical_method": f"Anonymous {action.lower()} method",
                "reason": "Synthetic generic method coverage.",
                "repair_parameters": {},
            }
        )
    copied["summary"] = {
        "features_analyzed": len(actions),
        "features_kept": 3,
        "features_repaired": 4,
        "features_dropped": 1,
    }
    validate_dashboard_run(copied)


def test_dashboard_builder_groups_technical_methods_without_feature_logic() -> None:
    expected = {
        "KEEP": ("KEEP", "No mitigation applied"),
        "KEEP_DRIFTED": ("KEEP", "Preserve same-semantic population drift"),
        "KEEP_SYSTEMIC_DRIFT": ("KEEP", "Preserve systemic drift"),
        "REALIGN_SCALE": ("REPAIR", "Scale realignment"),
        "REALIGN_OFFSET": ("REPAIR", "Offset realignment"),
        "REMAP_CATEGORIES": ("REPAIR", "Category remapping"),
        "REVERSE_PERCENTILE": ("REPAIR", "Reverse percentile representation"),
        "DROP_NOVEL_UNRELIABLE": ("DROP", "Exclude unreliable feature"),
    }
    assert {action: _decision_metadata(action) for action in expected} == expected


def test_missing_real_artifact_has_no_fallback(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Data unavailable for this run"):
        load_dashboard_run(tmp_path / "missing.json")


def test_dashboard_source_has_no_fabrication_paths() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "dashboard").rglob("*.py"))
    )
    prohibited = [
        "precision_recall_curve",
        "post_score -",
        "pre_mitigation_score",
        "mock_data",
        "fake_timeline",
        "synthetic_curve",
    ]
    for needle in prohibited:
        assert needle not in source
