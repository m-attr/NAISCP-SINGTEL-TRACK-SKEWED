from __future__ import annotations

import numpy as np
import pandas as pd

from common.contracts import POOLED_MONTH, TIME_COLUMN, canonicalize_category
from common.types import PreparationPlan
from preprocessing.rank_maps import apply_rank_map


def _map_frame(
    frame: pd.DataFrame,
    plan: PreparationPlan,
    split: str,
) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    rank_maps = plan.train_rank_maps if split == "train" else plan.test_rank_maps
    months = frame[TIME_COLUMN].astype("string")
    output: dict[str, np.ndarray] = {}

    for feature in plan.feature_order:
        if feature in plan.categorical_features:
            values = canonicalize_category(frame[feature])
            if split == "test" and feature in plan.category_remaps:
                values = values.replace(plan.category_remaps[feature])
            output[feature] = (
                values.map(plan.category_maps[feature])
                .fillna(-1)
                .astype(np.int32)
                .to_numpy()
            )
            continue

        if feature in plan.aligned_features:
            transformed = np.full(len(frame), np.nan, dtype=np.float32)
            feature_maps = rank_maps[feature]
            pooled = feature_maps[POOLED_MONTH]
            for month in months.unique().tolist():
                mask = (months == month).to_numpy()
                if not mask.any():
                    continue
                rank_map = feature_maps.get(str(month), pooled)
                transformed[mask] = apply_rank_map(
                    frame.loc[mask, feature], rank_map
                )
            if split == "test" and feature in plan.reversed_features:
                transformed = 1.0 - transformed
            fill_value = np.float32(plan.numeric_fill_values[feature])
            output[feature] = np.where(
                np.isnan(transformed), fill_value, transformed
            ).astype(np.float32)
            continue

        numeric = pd.to_numeric(frame[feature], errors="coerce")
        if split == "test" and feature in plan.scale_factors:
            numeric = numeric / plan.scale_factors[feature]
        elif split == "test" and feature in plan.offsets:
            numeric = numeric - plan.offsets[feature]
        output[feature] = (
            numeric.fillna(plan.numeric_fill_values[feature])
            .astype(np.float32)
            .to_numpy()
        )

    return pd.DataFrame(output, columns=plan.feature_order)


def transform_training(
    frame: pd.DataFrame,
    plan: PreparationPlan,
) -> pd.DataFrame:
    return _map_frame(frame, plan, split="train")


def transform_test(
    frame: pd.DataFrame,
    plan: PreparationPlan,
) -> pd.DataFrame:
    return _map_frame(frame, plan, split="test")
