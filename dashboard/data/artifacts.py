from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_NAME = "datadrift.dashboard_run"
SUPPORTED_SCHEMA_MAJOR = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "dashboard_artifacts" / "dashboard_run.json"


class ArtifactError(ValueError):
    """Raised when a dashboard artifact is absent, invalid, or incompatible."""


def resolve_artifact_path() -> Path:
    configured = os.getenv("DATADRIFT_DASHBOARD_ARTIFACT")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_ARTIFACT_PATH


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ArtifactError(f"Dashboard artifact field {key!r} must be an object.")
    return value


def validate_dashboard_run(payload: dict[str, Any]) -> None:
    schema = _require_mapping(payload, "schema")
    if schema.get("name") != SCHEMA_NAME:
        raise ArtifactError(f"Unsupported dashboard schema {schema.get('name')!r}.")
    version = str(schema.get("version", ""))
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"Invalid dashboard schema version {version!r}.") from exc
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise ArtifactError(
            f"Dashboard schema {version!r} is incompatible with this application."
        )
    if payload.get("mode") != "REAL_RUN":
        raise ArtifactError("Only explicitly labelled REAL_RUN artifacts are supported.")

    source = _require_mapping(payload, "source")
    if source.get("test_target_read") is not False:
        raise ArtifactError("Artifact does not certify test-target isolation.")
    _require_mapping(payload, "summary")
    run = _require_mapping(payload, "run")
    _require_mapping(run, "metrics")

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ArtifactError("Dashboard artifact contains no feature evidence.")
    names: set[str] = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ArtifactError(f"Feature row {index} is not an object.")
        name = str(feature.get("feature", "")).strip()
        if not name or name in names:
            raise ArtifactError(f"Feature row {index} has an invalid or duplicate name.")
        names.add(name)
        if feature.get("type") not in {"numeric", "categorical"}:
            raise ArtifactError(f"Feature {name!r} has an unsupported type.")
        decision = _require_mapping(feature, "decision")
        if decision.get("action") not in {
            "KEEP",
            "KEEP_DRIFTED",
            "KEEP_SYSTEMIC_DRIFT",
            "DROP_NOVEL_UNRELIABLE",
            "REVERSE_PERCENTILE",
            "REALIGN_SCALE",
            "REALIGN_OFFSET",
            "REMAP_CATEGORIES",
        }:
            raise ArtifactError(f"Feature {name!r} has an unsupported action.")
        evidence = _require_mapping(feature, "evidence")
        history = evidence.get("historical_shifts")
        if not isinstance(history, list):
            raise ArtifactError(f"Feature {name!r} has no chronological history list.")
        distribution = _require_mapping(feature, "distribution")
        if distribution.get("kind") != feature.get("type"):
            raise ArtifactError(f"Feature {name!r} distribution type is inconsistent.")

    summary = payload["summary"]
    if int(summary.get("features_analyzed", -1)) != len(features):
        raise ArtifactError("Feature count does not match the feature evidence list.")


def load_dashboard_run(path: str | Path | None = None) -> dict[str, Any]:
    artifact_path = Path(path).resolve() if path is not None else resolve_artifact_path()
    if not artifact_path.is_file():
        raise ArtifactError(
            f"Data unavailable for this run. Expected a real artifact at {artifact_path}."
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read dashboard artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("Dashboard artifact root must be a JSON object.")
    validate_dashboard_run(payload)
    payload["_artifact_path"] = str(artifact_path)
    return payload


def feature_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in payload["features"]:
        evidence = feature["evidence"]
        decision = feature["decision"]
        rows.append(
            {
                "Feature": feature["feature"],
                "Type": str(feature["type"]).title(),
                "Status": str(feature["status"]).title(),
                "Current": float(evidence["current_shift"]),
                "Historical": float(evidence["historical_max_shift"]),
                "Severity": float(evidence["novelty_ratio"]),
                "Action": str(decision["display_action"]).title(),
            }
        )
    return pd.DataFrame(rows)


def feature_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for feature in payload["features"]:
        if feature["feature"] == name:
            return feature
    raise ArtifactError(f"Feature {name!r} is not present in this run.")
