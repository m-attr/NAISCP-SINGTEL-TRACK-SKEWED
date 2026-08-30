from __future__ import annotations

import numpy as np
import pandas as pd

from common.contracts import POOLED_MONTH, TIME_COLUMN, sorted_month_values
from common.types import RankMap
from detection.detector import RANK_MAP_MAX_POINTS


def make_rank_map(
    values: pd.Series,
    max_points: int = RANK_MAP_MAX_POINTS,
) -> RankMap:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    numeric.sort()
    count = int(numeric.size)
    if count == 0:
        return RankMap(
            np.array([0.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )

    unique, first_indices, frequencies = np.unique(
        numeric,
        return_index=True,
        return_counts=True,
    )
    average_percentiles = (first_indices + (frequencies - 1) / 2.0 + 1.0) / count

    if len(unique) > max_points:
        selected = np.unique(
            np.linspace(0, len(unique) - 1, max_points).round().astype(np.int64)
        )
        unique = unique[selected]
        average_percentiles = average_percentiles[selected]

    return RankMap(
        unique.astype(np.float64),
        average_percentiles.astype(np.float64),
    )


def apply_rank_map(values: pd.Series, rank_map: RankMap) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    transformed = np.interp(
        numeric,
        rank_map.values,
        rank_map.percentiles,
        left=float(rank_map.percentiles[0]),
        right=float(rank_map.percentiles[-1]),
    )
    transformed[np.isnan(numeric)] = np.nan
    return transformed.astype(np.float32)


def build_rank_maps(
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, RankMap]]:
    month_values = frame[TIME_COLUMN].astype("string")
    result: dict[str, dict[str, RankMap]] = {}
    months = sorted_month_values(month_values)
    for feature in features:
        feature_maps: dict[str, RankMap] = {
            POOLED_MONTH: make_rank_map(frame[feature])
        }
        for month in months:
            feature_maps[month] = make_rank_map(
                frame.loc[month_values == month, feature]
            )
        result[feature] = feature_maps
    return result
