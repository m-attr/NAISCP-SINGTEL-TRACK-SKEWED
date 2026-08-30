from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from common.contracts import (
    ID_COLUMN,
    MISSING_CATEGORY,
    TARGET_COLUMN,
    TIME_COLUMN,
    canonicalize_category,
    sorted_month_values,
)
from common.types import FeatureDrift

# Method-level safeguards only; none names a public feature.
CURRENT_SHIFT_MIN = 0.10
NOVELTY_MIN = 2.0
PREDICTIVE_STRENGTH_MIN = 0.06
SYSTEMIC_DRIFT_FRACTION_MIN = 0.40
NOVEL_MISSING_SHIFT_MIN = 0.10
REPRESENTATION_MIN_OBSERVATIONS = 200
REPRESENTATION_MIN_UNIQUE = 20
SHAPE_KS_MAX = 0.10
SCALE_FACTOR_MIN = 5.0
SCALE_ALIGNMENT_ERROR_MAX = 0.35
OFFSET_SPREAD_RATIO_MIN = 0.80
OFFSET_SPREAD_RATIO_MAX = 1.25
OFFSET_SPREAD_MULTIPLIER_MIN = 6.0
CATEGORY_REMAP_MAX_CARDINALITY = 25
CATEGORY_REMAP_SUPPORT_MAX = 0.15
CATEGORY_REMAP_FREQUENCY_TV_MAX = 0.08
CATEGORY_REMAP_MIN_FREQUENCY_GAP = 0.05
STABLE_ANCHOR_SHIFT_MAX = 0.05
STABLE_ANCHOR_VALIDATION_MIN = 0.01
MIN_ORIENTATION_ANCHORS = 8
REVERSE_TRAIN_ASSOC_MIN = 0.08
REVERSE_TEST_ASSOC_MIN = 0.05
REVERSE_NOVELTY_MIN = 2.5
REVERSE_PREDICTIVE_STRENGTH_MIN = 0.20
REVERSE_SUPPORT_RETENTION_MIN = 0.85
RANK_UNIQUE_MIN = 10
RANK_MAP_MAX_POINTS = 8192
KS_MAX_VALUES = 4096
_PROFILE_BINS = 10
_PROFILE_SMOOTHING = 200.0


def bounded_numeric(values: pd.Series, maximum: int = KS_MAX_VALUES) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(array) <= maximum:
        return array
    indices = np.linspace(0, len(array) - 1, maximum, dtype=np.int64)
    return array[indices]


def numeric_ks(left: pd.Series, right: pd.Series) -> float:
    a = bounded_numeric(left)
    b = bounded_numeric(right)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(ks_2samp(a, b, method="asymp").statistic)


def categorical_total_variation(left: pd.Series, right: pd.Series) -> float:
    a = left.value_counts(normalize=True, dropna=False)
    b = right.value_counts(normalize=True, dropna=False)
    index = a.index.union(b.index)
    return 0.5 * float(
        (a.reindex(index, fill_value=0.0) - b.reindex(index, fill_value=0.0))
        .abs()
        .sum()
    )


def _numeric_representation_evidence(
    recent: pd.Series,
    current: pd.Series,
) -> tuple[float | None, float | None, dict[str, float | bool | None]]:
    """Identify only conspicuous pure scale or offset changes from robust summaries."""
    reference = bounded_numeric(recent)
    observed = bounded_numeric(current)
    checks: dict[str, float | bool | None] = {
        "enough_observations": False,
        "shape_ks": 1.0,
        "spread_ratio": 0.0,
        "scale_alignment_error": None,
        "offset_alignment_error": None,
        "offset_spread_multiples": 0.0,
        "scale_repair_safe": False,
        "offset_repair_safe": False,
    }
    if (
        len(reference) < REPRESENTATION_MIN_OBSERVATIONS
        or len(observed) < REPRESENTATION_MIN_OBSERVATIONS
        or len(np.unique(reference)) < REPRESENTATION_MIN_UNIQUE
        or len(np.unique(observed)) < REPRESENTATION_MIN_UNIQUE
    ):
        return None, None, checks

    probabilities = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    reference_quantiles = np.quantile(reference, probabilities)
    observed_quantiles = np.quantile(observed, probabilities)
    reference_median = float(reference_quantiles[2])
    observed_median = float(observed_quantiles[2])
    reference_spread = float(reference_quantiles[3] - reference_quantiles[1])
    observed_spread = float(observed_quantiles[3] - observed_quantiles[1])
    if reference_spread <= 1e-10 or observed_spread <= 1e-10:
        return None, None, checks

    standardized_reference = (reference - reference_median) / reference_spread
    standardized_observed = (observed - observed_median) / observed_spread
    shape_ks = float(
        ks_2samp(standardized_reference, standardized_observed, method="asymp").statistic
    )
    spread_ratio = observed_spread / reference_spread
    scale_alignment_error = float(
        np.max(
            np.abs(
                observed_quantiles / spread_ratio - reference_quantiles
            )
        )
        / reference_spread
    )
    offset = observed_median - reference_median
    offset_alignment_error = float(
        np.max(np.abs(observed_quantiles - offset - reference_quantiles))
        / reference_spread
    )
    offset_spread_multiples = abs(offset) / reference_spread

    scale_is_large = bool(
        spread_ratio >= SCALE_FACTOR_MIN
        or spread_ratio <= 1.0 / SCALE_FACTOR_MIN
    )
    scale_safe = bool(
        shape_ks <= SHAPE_KS_MAX
        and scale_is_large
        and scale_alignment_error <= SCALE_ALIGNMENT_ERROR_MAX
    )
    offset_safe = bool(
        shape_ks <= SHAPE_KS_MAX
        and OFFSET_SPREAD_RATIO_MIN <= spread_ratio <= OFFSET_SPREAD_RATIO_MAX
        and offset_spread_multiples >= OFFSET_SPREAD_MULTIPLIER_MIN
        and offset_alignment_error <= SCALE_ALIGNMENT_ERROR_MAX
    )
    checks.update(
        {
            "enough_observations": True,
            "shape_ks": shape_ks,
            "spread_ratio": spread_ratio,
            "scale_alignment_error": scale_alignment_error,
            "offset_alignment_error": offset_alignment_error,
            "offset_spread_multiples": offset_spread_multiples,
            "scale_repair_safe": scale_safe,
            "offset_repair_safe": offset_safe,
        }
    )
    return (
        float(spread_ratio) if scale_safe else None,
        float(offset) if offset_safe else None,
        checks,
    )


def _categorical_remap_evidence(
    recent: pd.Series,
    current: pd.Series,
) -> tuple[dict[str, str], dict[str, float | bool]]:
    """Infer relabeling only when frequency ranks form an unambiguous bijection."""
    reference = canonicalize_category(recent)
    observed = canonicalize_category(current)
    reference = reference[reference != MISSING_CATEGORY]
    observed = observed[observed != MISSING_CATEGORY]
    reference_shares = reference.value_counts(normalize=True)
    observed_shares = observed.value_counts(normalize=True)
    known_support = float(observed.isin(set(reference_shares.index)).mean()) if len(observed) else 0.0
    checks: dict[str, float | bool] = {
        "known_category_support_ratio": known_support,
        "compatible_cardinality": False,
        "frequency_structure_tv": 1.0,
        "minimum_frequency_gap": 0.0,
        "category_remap_safe": False,
        "mapping_confidence": 0.0,
    }
    cardinality = len(reference_shares)
    compatible = bool(
        len(reference) >= REPRESENTATION_MIN_OBSERVATIONS
        and len(observed) >= REPRESENTATION_MIN_OBSERVATIONS
        and 2 <= cardinality <= CATEGORY_REMAP_MAX_CARDINALITY
        and cardinality == len(observed_shares)
        and known_support <= CATEGORY_REMAP_SUPPORT_MAX
    )
    checks["compatible_cardinality"] = compatible
    if not compatible:
        return {}, checks

    reference_order = reference_shares.sort_values(ascending=False, kind="stable")
    observed_order = observed_shares.sort_values(ascending=False, kind="stable")
    reference_values = reference_order.to_numpy(dtype=np.float64)
    observed_values = observed_order.to_numpy(dtype=np.float64)
    frequency_tv = 0.5 * float(np.abs(reference_values - observed_values).sum())
    reference_gaps = np.abs(np.diff(reference_values))
    observed_gaps = np.abs(np.diff(observed_values))
    minimum_gap = float(
        min(reference_gaps.min(initial=1.0), observed_gaps.min(initial=1.0))
    )
    safe = bool(
        frequency_tv <= CATEGORY_REMAP_FREQUENCY_TV_MAX
        and minimum_gap >= CATEGORY_REMAP_MIN_FREQUENCY_GAP
    )
    confidence = (
        min(1.0, minimum_gap / (2.0 * CATEGORY_REMAP_MIN_FREQUENCY_GAP))
        * max(0.0, 1.0 - frequency_tv / CATEGORY_REMAP_FREQUENCY_TV_MAX)
        if safe
        else 0.0
    )
    checks.update(
        {
            "frequency_structure_tv": frequency_tv,
            "minimum_frequency_gap": minimum_gap,
            "category_remap_safe": safe,
            "mapping_confidence": float(confidence),
        }
    )
    if not safe:
        return {}, checks
    mapping = {
        str(current_label): str(reference_label)
        for current_label, reference_label in zip(
            observed_order.index.tolist(),
            reference_order.index.tolist(),
        )
    }
    return mapping, checks


def safe_spearman(
    left: np.ndarray | pd.Series,
    right: np.ndarray | pd.Series,
) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return 0.0
    a = a[mask]
    b = b[mask]
    if float(np.ptp(a)) <= 1e-12 or float(np.ptp(b)) <= 1e-12:
        return 0.0
    result = float(spearmanr(a, b).statistic)
    return result if np.isfinite(result) else 0.0


def _numeric_predictive_strength(values: pd.Series, target: np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna().to_numpy()
    if int(mask.sum()) < 20 or int(numeric[mask].nunique(dropna=True)) < 2:
        return 0.0
    return abs(
        safe_spearman(numeric.to_numpy(dtype=np.float64)[mask], target[mask])
    )


def _categorical_predictive_strength(values: pd.Series, target: np.ndarray) -> float:
    minimum_count = max(30, int(len(values) * 0.0004))
    grouped = (
        pd.DataFrame({"value": values, "target": target})
        .groupby("value", observed=True)["target"]
        .agg(["mean", "count"])
    )
    grouped = grouped[grouped["count"] >= minimum_count]
    if len(grouped) < 2:
        return 0.0
    center = float(np.average(grouped["mean"], weights=grouped["count"]))
    return float(
        np.sqrt(
            np.average(
                np.square(grouped["mean"] - center),
                weights=grouped["count"],
            )
        )
    )


def _base_evidence(
    train: pd.DataFrame,
    test_features: pd.DataFrame,
    target: np.ndarray,
) -> tuple[list[FeatureDrift], list[str]]:
    train_months = sorted_month_values(train[TIME_COLUMN])
    if not train_months:
        raise ValueError("Training data contains no valid Month values.")

    block_width = 2 if len(train_months) >= 4 else 1
    blocks = [
        train_months[index : index + block_width]
        for index in range(0, len(train_months), block_width)
    ]
    month_text = train[TIME_COLUMN].astype("string")
    block_frames = [train.loc[month_text.isin(block)] for block in blocks]
    recent = block_frames[-1]
    enough_history = len(train_months) >= 8 and len(blocks) >= 4

    shared_features = [
        column
        for column in train.columns
        if column in test_features.columns
        and column not in {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    ]
    categorical_features = [
        feature
        for feature in shared_features
        if not pd.api.types.is_numeric_dtype(train[feature])
    ]
    canonical_train = {
        feature: canonicalize_category(train[feature])
        for feature in categorical_features
    }
    canonical_test = {
        feature: canonicalize_category(test_features[feature])
        for feature in categorical_features
    }

    rows: list[FeatureDrift] = []
    for feature in shared_features:
        numeric = pd.api.types.is_numeric_dtype(train[feature])
        if numeric:
            current = numeric_ks(recent[feature], test_features[feature])
            history = [
                numeric_ks(
                    block_frames[index][feature],
                    block_frames[index + 1][feature],
                )
                for index in range(len(block_frames) - 1)
            ]
            predictive = _numeric_predictive_strength(train[feature], target)
            kind = "numeric"
            known_category_support = 1.0
            scale_factor, offset, representation_checks = (
                _numeric_representation_evidence(
                    recent[feature],
                    test_features[feature],
                )
            )
            category_mapping: dict[str, str] = {}
        else:
            current = categorical_total_variation(
                canonical_train[feature].loc[recent.index],
                canonical_test[feature],
            )
            history = [
                categorical_total_variation(
                    canonical_train[feature].loc[block_frames[index].index],
                    canonical_train[feature].loc[block_frames[index + 1].index],
                )
                for index in range(len(block_frames) - 1)
            ]
            predictive = _categorical_predictive_strength(
                canonical_train[feature], target
            )
            kind = "categorical"
            recent_values = canonical_train[feature].loc[recent.index]
            known_values = set(recent_values.unique().tolist())
            known_category_support = float(
                canonical_test[feature].isin(known_values).mean()
            )
            category_mapping, representation_checks = _categorical_remap_evidence(
                recent[feature],
                test_features[feature],
            )
            scale_factor = None
            offset = None

        historical_max = max(history) if history else 0.0
        novelty = current / (historical_max + 0.01)
        recent_unique = int(recent[feature].nunique(dropna=True))
        test_unique = int(test_features[feature].nunique(dropna=True))
        test_non_missing = int(test_features[feature].notna().sum())
        observable_reference_support = max(1, min(recent_unique, test_non_missing))
        support_retention = min(1.0, test_unique / observable_reference_support)
        missing_shift = abs(
            float(recent[feature].isna().mean())
            - float(test_features[feature].isna().mean())
        )
        missing_history = [
            abs(
                float(block_frames[index][feature].isna().mean())
                - float(block_frames[index + 1][feature].isna().mean())
            )
            for index in range(len(block_frames) - 1)
        ]
        historical_missing = max(missing_history) if missing_history else 0.0
        missing_novelty = missing_shift / (historical_missing + 0.01)
        missingness_is_main_change = bool(
            missing_shift >= max(0.05, 0.5 * current)
            and missing_novelty >= NOVELTY_MIN
        )
        unusual_movement = bool(
            enough_history
            and current >= CURRENT_SHIFT_MIN
            and novelty >= NOVELTY_MIN
            and not missingness_is_main_change
        )
        is_distribution_candidate = bool(
            unusual_movement
            and predictive >= PREDICTIVE_STRENGTH_MIN
        )
        is_missing_candidate = bool(
            enough_history
            and missing_shift >= NOVEL_MISSING_SHIFT_MIN
            and missing_novelty >= NOVELTY_MIN
            and predictive >= PREDICTIVE_STRENGTH_MIN
        )
        rows.append(
            FeatureDrift(
                feature=feature,
                kind=kind,
                current_shift=float(current),
                historical_max_shift=float(historical_max),
                novelty_ratio=float(novelty),
                predictive_strength=float(predictive),
                unique_values=int(train[feature].nunique(dropna=True)),
                recent_unique_values=recent_unique,
                test_unique_values=test_unique,
                support_retention_ratio=float(support_retention),
                missing_shift=float(missing_shift),
                historical_missing_shift=float(historical_missing),
                missing_novelty_ratio=float(missing_novelty),
                known_category_support_ratio=float(known_category_support),
                unusual_movement=unusual_movement,
                scale_factor=scale_factor,
                offset=offset,
                category_mapping=category_mapping,
                evidence_checks={
                    **representation_checks,
                    "missingness_is_main_change": missingness_is_main_change,
                },
                action=(
                    "REVIEW_MISSINGNESS"
                    if is_missing_candidate
                    else "REVIEW_NOVEL"
                    if is_distribution_candidate
                    else "KEEP"
                ),
                reason=(
                    "The newest missingness movement is unusually large relative to this "
                    "feature's own training history."
                    if is_missing_candidate
                    else
                    "The newest unlabeled movement is unusually large relative to this "
                    "feature's own training history."
                    if is_distribution_candidate
                    else "No sufficiently strong evidence that intervention is safer than "
                    "keeping the feature."
                ),
            )
        )
    return rows, train_months


def _add_systemic_context(rows: list[FeatureDrift]) -> list[FeatureDrift]:
    """Mark broad movement by proportion so behavior is stable across dataset widths."""
    eligible = [row for row in rows if row.unique_values > 1]
    fraction = (
        sum(row.unusual_movement for row in eligible) / len(eligible)
        if eligible
        else 0.0
    )
    active = bool(fraction >= SYSTEMIC_DRIFT_FRACTION_MIN)
    return [
        replace(
            row,
            systemic_drift_fraction=float(fraction),
            systemic_drift_active=active,
        )
        for row in rows
    ]


def _build_training_label_rate_table(
    reference: pd.DataFrame,
    target: np.ndarray,
    feature: str,
    global_rate: float,
) -> tuple[str, np.ndarray | None, pd.Series] | None:
    """Build a deterministic training-label summary, never a fitted estimator."""
    if pd.api.types.is_numeric_dtype(reference[feature]):
        raw = pd.to_numeric(reference[feature], errors="coerce")
        non_missing = raw.dropna()
        if non_missing.nunique() < 3:
            return None
        edges = np.unique(
            non_missing.quantile(
                np.linspace(0.0, 1.0, _PROFILE_BINS + 1)
            ).to_numpy()
        )
        if len(edges) < 3:
            return None
        edges[0] = -np.inf
        edges[-1] = np.inf
        buckets = pd.cut(raw, edges, include_lowest=True, duplicates="drop")
        grouped = (
            pd.DataFrame({"bucket": buckets, "target": target})
            .groupby("bucket", observed=True)["target"]
            .agg(["sum", "count"])
        )
        rates = (grouped["sum"] + _PROFILE_SMOOTHING * global_rate) / (
            grouped["count"] + _PROFILE_SMOOTHING
        )
        return "numeric", edges, rates

    values = canonicalize_category(reference[feature])
    grouped = (
        pd.DataFrame({"value": values, "target": target})
        .groupby("value", observed=True)["target"]
        .agg(["sum", "count"])
    )
    rates = (grouped["sum"] + _PROFILE_SMOOTHING * global_rate) / (
        grouped["count"] + _PROFILE_SMOOTHING
    )
    return "categorical", None, rates


def _apply_training_label_rate_table(
    frame: pd.DataFrame,
    feature: str,
    profile: tuple[str, np.ndarray | None, pd.Series],
    global_rate: float,
) -> np.ndarray:
    kind, edges, rates = profile
    if kind == "numeric":
        assert edges is not None
        buckets = pd.cut(
            pd.to_numeric(frame[feature], errors="coerce"),
            edges,
            include_lowest=True,
            duplicates="drop",
        )
        return buckets.map(rates).astype(float).fillna(global_rate).to_numpy()
    return (
        canonicalize_category(frame[feature])
        .map(rates)
        .astype(float)
        .fillna(global_rate)
        .to_numpy()
    )


def _add_orientation_evidence(
    train: pd.DataFrame,
    test_features: pd.DataFrame,
    target: np.ndarray,
    rows: list[FeatureDrift],
    train_months: list[str],
) -> list[FeatureDrift]:
    candidates = {row.feature for row in rows if row.action == "REVIEW_NOVEL"}
    if not candidates:
        return rows

    recent_months = train_months[-2:] if len(train_months) >= 4 else train_months[-1:]
    month_text = train[TIME_COLUMN].astype("string")
    recent = train.loc[month_text.isin(recent_months)]
    reference = train.loc[~month_text.isin(recent_months)]
    target_series = pd.Series(np.asarray(target), index=train.index)
    reference_target = target_series.loc[reference.index].to_numpy()
    recent_target = target_series.loc[recent.index].to_numpy()
    global_rate = float(reference_target.mean())
    row_map = {row.feature: row for row in rows}

    anchors: list[tuple[np.ndarray, np.ndarray, float]] = []
    for feature in train.columns:
        if (
            feature in {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
            or feature in candidates
            or feature not in test_features.columns
        ):
            continue
        drift = row_map[feature]
        if (
            drift.current_shift > STABLE_ANCHOR_SHIFT_MAX
            or drift.novelty_ratio >= NOVELTY_MIN
        ):
            continue
        profile = _build_training_label_rate_table(
            reference, reference_target, feature, global_rate
        )
        if profile is None:
            continue
        recent_scores = _apply_training_label_rate_table(
            recent, feature, profile, global_rate
        )
        test_scores = _apply_training_label_rate_table(
            test_features, feature, profile, global_rate
        )
        validation = safe_spearman(recent_scores, recent_target)
        if abs(validation) < STABLE_ANCHOR_VALIDATION_MIN:
            continue
        center = float(np.mean(recent_scores))
        spread = float(np.std(recent_scores))
        if spread <= 1e-8:
            continue
        anchors.append(
            (
                (recent_scores - center) / spread,
                (test_scores - center) / spread,
                abs(validation),
            )
        )

    anchors.sort(key=lambda item: item[2], reverse=True)
    anchors = anchors[:32]
    anchor_count = len(anchors)
    if anchor_count >= MIN_ORIENTATION_ANCHORS:
        recent_consensus = np.mean(np.vstack([item[0] for item in anchors]), axis=0)
        test_consensus = np.mean(np.vstack([item[1] for item in anchors]), axis=0)
    else:
        recent_consensus = np.array([], dtype=np.float64)
        test_consensus = np.array([], dtype=np.float64)

    output: list[FeatureDrift] = []
    for row in rows:
        if row.feature not in candidates:
            output.append(row)
            continue

        orientation_train = 0.0
        orientation_test = 0.0
        if (
            row.kind == "numeric"
            and row.unique_values >= RANK_UNIQUE_MIN
            and anchor_count >= MIN_ORIENTATION_ANCHORS
        ):
            recent_rank = recent.groupby(TIME_COLUMN, observed=True)[row.feature].rank(
                pct=True, method="average"
            )
            test_rank = test_features.groupby(TIME_COLUMN, observed=True)[row.feature].rank(
                pct=True, method="average"
            )
            orientation_train = safe_spearman(recent_rank, recent_consensus)
            orientation_test = safe_spearman(test_rank, test_consensus)

        output.append(
            replace(
                row,
                orientation_train=(
                    orientation_train if np.isfinite(orientation_train) else 0.0
                ),
                orientation_test=(
                    orientation_test if np.isfinite(orientation_test) else 0.0
                ),
                orientation_anchor_count=anchor_count,
            )
        )
    return output


def detect_drift(
    train: pd.DataFrame,
    test_features: pd.DataFrame,
    target: np.ndarray,
) -> list[FeatureDrift]:
    """Measure shift, historical surprise, usefulness, missingness, and orientation."""
    evidence, train_months = _base_evidence(train, test_features, target)
    evidence = _add_systemic_context(evidence)
    return _add_orientation_evidence(
        train,
        test_features,
        target,
        evidence,
        train_months,
    )
