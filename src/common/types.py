from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np

from common.contracts import ID_COLUMN, TARGET_COLUMN, TIME_COLUMN


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    kind: str
    current_shift: float
    historical_max_shift: float
    novelty_ratio: float
    predictive_strength: float
    unique_values: int
    recent_unique_values: int
    test_unique_values: int
    support_retention_ratio: float
    missing_shift: float
    historical_missing_shift: float = 0.0
    missing_novelty_ratio: float = 0.0
    known_category_support_ratio: float = 1.0
    unusual_movement: bool = False
    systemic_drift_fraction: float = 0.0
    systemic_drift_active: bool = False
    scale_factor: float | None = None
    offset: float | None = None
    category_mapping: dict[str, str] = field(default_factory=dict)
    evidence_checks: dict[str, Any] = field(default_factory=dict)
    orientation_train: float = 0.0
    orientation_test: float = 0.0
    orientation_anchor_count: int = 0
    action: str = "KEEP"
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "current_shift": self.current_shift,
            "historical_max_shift": self.historical_max_shift,
            "novelty_ratio": self.novelty_ratio,
            "predictive_strength": self.predictive_strength,
            "unique_values": self.unique_values,
            "recent_unique_values": self.recent_unique_values,
            "test_unique_values": self.test_unique_values,
            "support_retention_ratio": self.support_retention_ratio,
            "missing_shift": self.missing_shift,
            "historical_missing_shift": self.historical_missing_shift,
            "missing_novelty_ratio": self.missing_novelty_ratio,
            "known_category_support_ratio": self.known_category_support_ratio,
            "unusual_movement": self.unusual_movement,
            "systemic_drift_fraction": self.systemic_drift_fraction,
            "systemic_drift_active": self.systemic_drift_active,
            "scale_factor": self.scale_factor,
            "offset": self.offset,
            "category_mapping": dict(self.category_mapping),
            "evidence_checks": dict(self.evidence_checks),
            "orientation_train": self.orientation_train,
            "orientation_test": self.orientation_test,
            "orientation_anchor_count": self.orientation_anchor_count,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RankMap:
    values: np.ndarray
    percentiles: np.ndarray


@dataclass
class PreparationPlan:
    feature_order: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    dropped_features: list[str]
    aligned_features: list[str]
    reversed_features: list[str]
    category_maps: dict[str, dict[str, int]]
    numeric_fill_values: dict[str, float]
    train_rank_maps: dict[str, dict[str, RankMap]]
    test_rank_maps: dict[str, dict[str, RankMap]]
    drift: list[FeatureDrift]
    scale_factors: dict[str, float] = field(default_factory=dict)
    offsets: dict[str, float] = field(default_factory=dict)
    category_remaps: dict[str, dict[str, str]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def ranked_features(self) -> list[str]:
        """Features represented by stored percentile maps."""
        return self.aligned_features

    def summary(self) -> dict[str, Any]:
        return {
            "contract_columns": {
                "id": ID_COLUMN,
                "time": TIME_COLUMN,
                "target": TARGET_COLUMN,
            },
            "feature_order": self.feature_order,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "dropped_features": self.dropped_features,
            "aligned_features": self.aligned_features,
            "reversed_features": self.reversed_features,
            "scale_factors": self.scale_factors,
            "offsets": self.offsets,
            "category_remap_cardinality": {
                feature: len(mapping)
                for feature, mapping in self.category_remaps.items()
            },
            "category_remaps": self.category_remaps,
            "category_cardinality": {
                feature: len(mapping) for feature, mapping in self.category_maps.items()
            },
            "rank_map_points": {
                split: {
                    feature: {
                        month: int(len(rank_map.values))
                        for month, rank_map in months.items()
                    }
                    for feature, months in maps.items()
                }
                for split, maps in (
                    ("train", self.train_rank_maps),
                    ("test", self.test_rank_maps),
                )
            },
            "thresholds": self.thresholds,
            "drift": [row.as_dict() for row in self.drift],
        }

    def save_summary(self, path: str | Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, ensure_ascii=False)
