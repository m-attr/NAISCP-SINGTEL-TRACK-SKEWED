import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from sklearn.preprocessing import OrdinalEncoder
import re
from itertools import combinations
from schema_utils import name_tokens
from utils import _env_truthy


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert pandas to polars using Arrow to avoid multiprocessing issues on Windows.
    """
    try:
        return pl.from_arrow(pa.Table.from_pandas(df, preserve_index=False))
    except Exception:
        # Fallback path for mixed dtypes.
        series = [pl.Series(name=c, values=df[c].tolist(), strict=False) for c in df.columns]
        return pl.DataFrame(series)


def _safe_feature_name(*parts: str) -> str:
    raw = "__".join(str(p) for p in parts if p is not None and str(p) != "")
    return re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_")


def _is_identifier_name(name: str) -> bool:
    # REFACTORED: reuse shared schema token parser.
    tokens = name_tokens(name)
    if not tokens:
        return False
    if "id" in tokens:
        return True
    return any(tok in {"uuid", "guid", "identifier", "key"} for tok in tokens)


def _is_target_name(name: str) -> bool:
    # REFACTORED: reuse shared schema token parser.
    tokens = name_tokens(name)
    return any(tok in {"target", "label", "class", "outcome", "response"} for tok in tokens)


def _detect_duration_like_column(df: pd.DataFrame) -> str | None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return None

    best_col = None
    best_score = -1.0

    for col in numeric_cols:
        if _is_identifier_name(col) or _is_target_name(col):
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 100:
            continue

        nunique = int(s.nunique(dropna=True))
        if nunique < 3:
            continue

        arr = s.to_numpy(dtype=np.float64, copy=False)
        nonneg_ratio = float((arr >= 0).mean())
        int_like_ratio = float(np.isclose(np.mod(arr, 1.0), 0.0, atol=1e-8).mean())
        zero_ratio = float((arr == 0.0).mean())
        q95 = float(np.quantile(arr, 0.95))
        q05 = float(np.quantile(arr, 0.05))
        spread = max(1e-6, q95 - q05)

        score = 0.0
        if int_like_ratio >= 0.90:
            score += 1.5
        if int_like_ratio >= 0.90:
            score += 0.5
        if nonneg_ratio >= 0.95:
            score += 1.0
        if zero_ratio <= 0.40:
            score += 0.5
        if q95 <= 500.0:
            score += 1.0
        score += float(min(1.5, np.log1p(nunique) / 4.0))
        score += float(max(0.0, 1.0 - np.log1p(spread) / 8.0))

        if score > best_score:
            best_score = score
            best_col = col

    return best_col if best_score >= 1.5 else None


def _detect_amount_like_columns(df: pd.DataFrame, duration_col: str | None = None, max_cols: int = 8) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    scored: list[tuple[str, float]] = []

    for col in numeric_cols:
        if col == duration_col or _is_identifier_name(col) or _is_target_name(col):
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 100:
            continue

        arr = s.to_numpy(dtype=np.float64, copy=False)
        if np.isnan(arr).all():
            continue

        nonneg_ratio = float((arr >= 0).mean())
        nunique = int(s.nunique(dropna=True))
        std = float(np.nanstd(arr))
        iqr = float(np.nanquantile(arr, 0.75) - np.nanquantile(arr, 0.25))

        score = 0.0
        score += float(min(2.5, np.log1p(max(0.0, std))))
        score += float(min(1.5, np.log1p(max(0.0, iqr))))
        score += float(min(1.0, np.log1p(max(1, nunique)) / 5.0))
        if nonneg_ratio >= 0.80:
            score += 0.75
        if nunique >= 20 and std > 0.0:
            scored.append((col, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_cols]]


def _select_spend_velocity_pair(
    df: pd.DataFrame,
    duration_col: str,
    amount_cols: list[str],
    min_corr: float = 0.92,
    min_samples: int = 200,
) -> tuple[str, str] | None:
    """
    Find financial columns A and B where (A * duration) strongly tracks B.
    """
    if duration_col not in df.columns or len(amount_cols) < 2:
        return None

    d = pd.to_numeric(df[duration_col], errors="coerce")
    if d.notna().sum() < min_samples:
        return None

    best_pair: tuple[str, str] | None = None
    best_corr = float(min_corr)

    for a_col in amount_cols:
        if a_col not in df.columns:
            continue
        a = pd.to_numeric(df[a_col], errors="coerce")
        ad = a * d
        for b_col in amount_cols:
            if b_col == a_col or b_col not in df.columns:
                continue
            b = pd.to_numeric(df[b_col], errors="coerce")
            tmp = pd.concat([ad, b], axis=1).dropna()
            if len(tmp) < min_samples:
                continue
            corr = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))
            spear = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="spearman"))
            robust_corr = min(corr, spear) if np.isfinite(spear) else corr
            if np.isfinite(robust_corr) and robust_corr > best_corr:
                best_corr = robust_corr
                best_pair = (a_col, b_col)

    return best_pair


def _select_ratio_pairs_by_correlation(
    df: pd.DataFrame,
    max_pairs: int = 6,
    min_corr: float = 0.60,
    max_candidate_cols: int = 28,
) -> list[tuple[str, str]]:
    """
    Select ratio pairs from numeric columns using correlation structure only.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns.tolist()
        if not _is_identifier_name(c) and not _is_target_name(c)
    ]
    if len(numeric_cols) < 2:
        return []

    sampled = df[numeric_cols]
    if len(sampled) > 120000:
        sampled = sampled.sample(n=120000, random_state=42)

    score_table: list[tuple[str, float]] = []
    for col in numeric_cols:
        s = pd.to_numeric(sampled[col], errors="coerce")
        nunique = int(s.nunique(dropna=True))
        if nunique < 10:
            continue
        std = float(s.std(skipna=True))
        if not np.isfinite(std) or std <= 0.0:
            continue
        score_table.append((col, float(np.log1p(std) + np.log1p(nunique) / 5.0)))

    if len(score_table) < 2:
        return []

    score_table.sort(key=lambda x: x[1], reverse=True)
    candidate_cols = [c for c, _ in score_table[:max_candidate_cols]]
    corr = sampled[candidate_cols].corr(method="spearman").abs().fillna(0.0)

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(candidate_cols)):
        for j in range(i + 1, len(candidate_cols)):
            c1 = candidate_cols[i]
            c2 = candidate_cols[j]
            val = float(corr.loc[c1, c2])
            if val >= min_corr:
                pairs.append((c1, c2, val))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[:max_pairs]]


def _detect_suffix_numeric_groups(
    df: pd.DataFrame,
    max_groups: int = 8,
    max_group_size: int = 10,
) -> list[tuple[str, list[str]]]:
    """
    Group numeric columns by shared prefix + ordered numeric suffix patterns
    like foo_1, foo_2, foo_3 or foo_m1, foo_m2.
    """
    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns.tolist()
        if not _is_identifier_name(c) and not _is_target_name(c)
    ]
    if len(num_cols) < 2:
        return []

    groups: dict[str, list[tuple[int, str]]] = {}
    pat = re.compile(r"^(.*?)(?:[_\-]?m)?(\d+)$", flags=re.IGNORECASE)

    for col in num_cols:
        m = pat.match(str(col))
        if not m:
            continue
        prefix = re.sub(r"[_\-]+$", "", m.group(1).strip())
        if not prefix:
            continue
        idx = int(m.group(2))
        groups.setdefault(prefix, []).append((idx, col))

    out: list[tuple[str, list[str]]] = []
    for prefix, members in groups.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda x: x[0])

        # Keep the longest contiguous sequence to avoid noisy mixed suffix sets.
        best_seq = [members_sorted[0]]
        seq = [members_sorted[0]]
        for curr in members_sorted[1:]:
            if curr[0] == seq[-1][0] + 1:
                seq.append(curr)
            else:
                if len(seq) > len(best_seq):
                    best_seq = seq
                seq = [curr]
        if len(seq) > len(best_seq):
            best_seq = seq

        if len(best_seq) >= 3:
            cols = [c for _, c in best_seq[:max_group_size]]
            out.append((prefix, cols))

    out.sort(key=lambda x: len(x[1]), reverse=True)
    return out[:max_groups]


def _select_interaction_categorical_columns(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    max_cols: int = 5,
    max_cardinality: int = 40,
) -> list[str]:
    exclude = exclude_cols or set()
    scored: list[tuple[str, float]] = []

    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for col in cat_cols:
        if col in exclude or _is_identifier_name(col) or _is_target_name(col):
            continue

        s = df[col]
        nunique = int(s.nunique(dropna=True))
        if nunique < 2 or nunique > max_cardinality:
            continue

        missing_rate = float(s.isna().mean())
        if missing_rate > 0.75:
            continue

        probs = s.value_counts(normalize=True, dropna=True).to_numpy(dtype=np.float64, copy=False)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum()) if probs.size else 0.0
        entropy_norm = entropy / max(1.0, np.log(max(2, nunique)))

        score = float(nunique) * (1.0 - 0.5 * missing_rate) + 2.0 * entropy_norm
        if nunique == 2:
            score -= 0.25
        scored.append((col, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_cols]]


def _derive_time_index(series: pd.Series) -> pd.Series:
    month_idx = _extract_month_index(series)
    if float(month_idx.notna().mean()) >= 0.40:
        return month_idx.astype("float32")

    parsed = pd.to_datetime(series.astype("string"), errors="coerce")
    if float(parsed.notna().mean()) >= 0.40:
        return parsed.dt.month.astype("float32")

    return pd.Series(np.nan, index=series.index, dtype="float32")


def _detect_time_like_column(df: pd.DataFrame) -> str | None:
    text_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if not text_cols:
        return None

    enhanced_time_detection = _env_truthy("ENABLE_ENHANCED_TIME_DETECTION", "0")

    best_col = None
    best_score = -1.0

    for col in text_cols:
        if _is_identifier_name(col) or _is_target_name(col):
            continue

        s = df[col].dropna().astype("string")
        if s.empty:
            continue
        s = s.iloc[: min(len(s), 4000)]

        try:
            dt_parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        except Exception:
            dt_parsed = pd.to_datetime(s, errors="coerce")
        dt_ratio = float(dt_parsed.notna().mean())
        nunique = int(s.nunique(dropna=True))
        cardinality_penalty = 0.0 if nunique <= 1000 else min(0.3, np.log1p(nunique - 1000) / 10.0)

        if enhanced_time_detection:
            # Robust mode for compact month formats (e.g., '25-Jan') plus time-like names.
            month_ratio = float(_extract_month_index(s).notna().mean())
            tokens = set(name_tokens(col))
            temporal_tokens = {"month", "date", "time", "period", "timestamp", "day", "week", "year"}
            name_bonus = 0.0
            if tokens & temporal_tokens:
                name_bonus += 0.20
            if "month" in tokens:
                name_bonus += 0.10
            score = max(dt_ratio, month_ratio) + name_bonus - float(cardinality_penalty)
        else:
            score = dt_ratio - float(cardinality_penalty)

        if score > best_score:
            best_score = score
            best_col = col

    threshold = 0.30 if enhanced_time_detection else 0.45
    return best_col if best_score >= threshold else None


def _detect_binary_positive_text_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect binary-positive text columns from observed value patterns (dtype-driven).
    """
    positive_values = {"yes", "true", "1", "y", "on", "enabled"}
    negative_values = {"no", "false", "0", "n", "off", "disabled"}
    out: list[str] = []
    for col in df.select_dtypes(include=["object", "category", "string"]).columns.tolist():
        if _is_identifier_name(col) or _is_target_name(col):
            continue

        s = df[col].dropna().astype("string").str.strip().str.lower()
        if s.empty:
            continue
        s = s.iloc[: min(len(s), 200000)]
        values = set(s.unique().tolist())
        if 2 <= len(values) <= 6 and (values & positive_values) and (values & negative_values):
            out.append(col)
    return out


def _detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Dynamically detect latitude/longitude-like numeric columns by value ranges.
    """
    lat_candidates: list[str] = []
    lon_candidates: list[str] = []
    for col in df.select_dtypes(include=["number"]).columns.tolist():
        if _is_identifier_name(col) or _is_target_name(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 100:
            continue
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        nunique = int(s.nunique(dropna=True))
        if nunique < 50:
            continue
        if lo >= -90.0 and hi <= 90.0:
            lat_candidates.append(col)
        elif lo >= -180.0 and hi <= 180.0:
            lon_candidates.append(col)

    lat_col = lat_candidates[0] if lat_candidates else None
    lon_col = lon_candidates[0] if lon_candidates else None
    if lat_col == lon_col:
        lon_col = lon_candidates[1] if len(lon_candidates) > 1 else None
    return lat_col, lon_col


def _extract_month_index(series: pd.Series) -> pd.Series:
    """
    Map free-form month strings (e.g., '25-jan', 'jan', '2025-jan') to 1..12.
    """
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    normalized = (
        series.astype("string")
        .str.lower()
        .str.extract(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", expand=False)
    )
    return normalized.map(month_map).astype("float32")


def clean_categories(df: pd.DataFrame):
    """
    Fix case-sensitivity issues: lowercase everything in the dataset.
    """
    category_columns = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if not _is_identifier_name(c)
    ]
    if not category_columns:
        return df

    time_col = _detect_time_like_column(df)

    pl_df = _to_polars(df)
    exprs = []
    for column in category_columns:
        base = (
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
        )
        if column != time_col:
            base = (
                base
                .str.replace_all(r"[_\-]+", " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )
        exprs.append(base.alias(column))
    return pl_df.with_columns(exprs).to_pandas()


def _apply_percentile_rank_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous numeric features to percentile ranks (0..1).
    This reduces sensitivity to global shifts in magnitude across datasets.
    """
    if df.empty:
        return df

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        col_lower = col.lower()
        if "id" in col_lower or col_lower.endswith("code"):
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        nunique = int(series.nunique(dropna=True))
        if nunique <= 20:
            # Keep low-card numeric columns in original scale.
            continue
        # Preserve sparsity semantics for heavily zero-inflated numerics.
        zero_rate = float((series == 0).mean())
        if zero_rate >= 0.60:
            continue

        ranked = series.rank(pct=True, method="average")
        df[col] = ranked.astype(np.float32)
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run category normalization + feature engineering in one Polars lazy pipeline,
    then collect once back to pandas.
    """
    if df.empty:
        return df

    enable_advanced_dynamic = _env_truthy("ENABLE_ADV_DYNAMIC_FEATURES", "1")
    enable_global_inactivity = _env_truthy("ENABLE_GLOBAL_INACTIVITY_INDEX", "0")
    enable_spend_velocity = _env_truthy("ENABLE_DYNAMIC_SPEND_VELOCITY", "0")
    enable_suffix_delta = _env_truthy("ENABLE_SUFFIX_DELTA_GROUPING", "0")

    category_columns = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if not _is_identifier_name(c)
    ]
    duration_col = _detect_duration_like_column(df)
    amount_cols = _detect_amount_like_columns(df, duration_col=duration_col, max_cols=8)
    time_col = _detect_time_like_column(df)
    spend_velocity_pair = (
        _select_spend_velocity_pair(df, duration_col, amount_cols)
        if enable_advanced_dynamic and enable_spend_velocity and duration_col
        else None
    )
    ratio_pairs = _select_ratio_pairs_by_correlation(df, max_pairs=6, min_corr=0.60)
    suffix_groups = _detect_suffix_numeric_groups(df) if (enable_advanced_dynamic and enable_suffix_delta) else []
    interaction_cats = _select_interaction_categorical_columns(
        df,
        exclude_cols={time_col} if time_col else set(),
        max_cols=5,
        max_cardinality=25,
    )

    lf = _to_polars(df).lazy()
    exprs = []
    per_duration_pairs: list[tuple[str, str]] = []

    if category_columns:
        for column in category_columns:
            base = (
                pl.col(column)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.to_lowercase()
            )
            if column != time_col:
                base = (
                    base
                    .str.replace_all(r"[_\-]+", " ")
                    .str.replace_all(r"\s+", " ")
                    .str.strip_chars()
                )
            exprs.append(base.alias(column))

    positive_binary_cols = _detect_binary_positive_text_columns(df)
    if positive_binary_cols:
        binary_exprs = [
            (
                pl.col(c)
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .eq("yes")
                .cast(pl.Int8)
            )
            for c in positive_binary_cols
        ]
        exprs.append(
            pl.sum_horizontal(binary_exprs)
            .cast(pl.Int8)
            .alias("Total_Binary_Positive_Count")
        )
    else:
        exprs.append(pl.lit(0, dtype=pl.Int8).alias("Total_Binary_Positive_Count"))

    lat_col, lon_col = _detect_lat_lon_columns(df)
    if lat_col and lon_col:
        exprs.append(pl.col(lat_col).round(1).alias("Lat_Grid"))
        exprs.append(pl.col(lon_col).round(1).alias("Lon_Grid"))

    if enable_advanced_dynamic:
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns.tolist()
            if not _is_identifier_name(c) and not _is_target_name(c)
        ]
        if numeric_cols and enable_global_inactivity:
            inactivity_terms = [
                pl.when(pl.col(c).is_null() | (pl.col(c).cast(pl.Float64, strict=False) == 0.0))
                .then(1.0)
                .otherwise(0.0)
                for c in numeric_cols
            ]
            exprs.append(
                (
                    pl.sum_horizontal(inactivity_terms)
                    / pl.lit(float(len(inactivity_terms)), dtype=pl.Float32)
                ).cast(pl.Float32).alias("Global_Inactivity_Index")
            )

        if spend_velocity_pair and duration_col:
            a_col, b_col = spend_velocity_pair
            exprs.append(
                (
                    (pl.col(a_col).cast(pl.Float32) * pl.col(duration_col).cast(pl.Float32))
                    / (pl.col(b_col).cast(pl.Float32).abs() + pl.lit(1e-5, dtype=pl.Float32))
                ).alias("Dynamic_Spend_Velocity")
            )

    # Ratio primitives from correlated numeric pairs.
    for idx, (num_col, den_col) in enumerate(ratio_pairs, start=1):
        exprs.append(
            (
                pl.col(num_col).cast(pl.Float32)
                / (pl.col(den_col).cast(pl.Float32).abs() + pl.lit(1e-5, dtype=pl.Float32))
            ).alias(f"AutoRatio_{idx}")
        )

    # Dynamic ratio features: amount-like columns normalized by a duration-like column.
    if duration_col and duration_col in df.columns:
        for col in amount_cols[:6]:
            alias = _safe_feature_name(col, "per", duration_col)
            per_duration_pairs.append((col, alias))
            exprs.append(
                (
                    pl.col(col).cast(pl.Float32)
                    / (pl.col(duration_col).cast(pl.Float32).abs() + pl.lit(1.0, dtype=pl.Float32))
                ).alias(alias)
            )

    # Dynamic share features: high-correlation amount columns as share of a base amount column.
    base_amount_col = amount_cols[0] if amount_cols else None
    if base_amount_col:
        component_cols: list[str] = []
        base_series = pd.to_numeric(df[base_amount_col], errors="coerce")
        for c in amount_cols[1:]:
            if c not in df.columns:
                continue
            candidate = pd.to_numeric(df[c], errors="coerce")
            tmp = pd.concat([base_series, candidate], axis=1).dropna()
            if len(tmp) < 200:
                continue
            corr = float(abs(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="spearman")))
            if np.isfinite(corr) and corr >= 0.30:
                component_cols.append(c)

        for col in component_cols[:4]:
            alias = _safe_feature_name(col, "share", base_amount_col)
            exprs.append(
                (
                    pl.col(col).cast(pl.Float32)
                    / (pl.col(base_amount_col).cast(pl.Float32).abs() + pl.lit(1.0, dtype=pl.Float32))
                ).alias(alias)
            )

    # Dynamic categorical interactions from low-cardinality text segments.
    pair_count = 0
    for c1, c2 in combinations(interaction_cats[:4], 2):
        pair_count += 1
        exprs.append(
            pl.concat_str(
                [
                    pl.col(c1).cast(pl.Utf8, strict=False).fill_null("missing"),
                    pl.lit("|"),
                    pl.col(c2).cast(pl.Utf8, strict=False).fill_null("missing"),
                ],
                ignore_nulls=False,
            ).alias(f"CatPair_{pair_count}")
        )
        if pair_count >= 4:
            break

    if len(interaction_cats) >= 3:
        c1, c2, c3 = interaction_cats[:3]
        exprs.append(
            pl.concat_str(
                [
                    pl.col(c1).cast(pl.Utf8, strict=False).fill_null("missing"),
                    pl.lit("|"),
                    pl.col(c2).cast(pl.Utf8, strict=False).fill_null("missing"),
                    pl.lit("|"),
                    pl.col(c3).cast(pl.Utf8, strict=False).fill_null("missing"),
                ],
                ignore_nulls=False,
            ).alias("CatTriple_1")
        )

    out = lf.with_columns(exprs).collect(streaming=True).to_pandas()

    if enable_advanced_dynamic and suffix_groups:
        for prefix, cols in suffix_groups:
            cols = [c for c in cols if c in out.columns]
            if len(cols) < 2:
                continue
            values = out[cols].apply(pd.to_numeric, errors="coerce")
            max_delta_name = _safe_feature_name(prefix, "Max_Delta")
            volatility_name = _safe_feature_name(prefix, "Volatility")
            out[max_delta_name] = (values.max(axis=1, skipna=True) - values.min(axis=1, skipna=True)).astype(np.float32)
            out[volatility_name] = values.std(axis=1, skipna=True, ddof=0).astype(np.float32)

    # Dynamic duration segmentation and cross-segment interactions.
    if duration_col and duration_col in out.columns:
        duration_vals = pd.to_numeric(out[duration_col], errors="coerce")
        valid_duration = duration_vals.dropna()
        if len(valid_duration) > 100:
            q = np.unique(np.quantile(valid_duration.to_numpy(dtype=np.float64, copy=False), [0.0, 0.15, 0.35, 0.60, 0.85, 1.0]))
            if len(q) >= 3:
                labels = [f"d{i+1}" for i in range(len(q) - 1)]
                out["DurationBand"] = pd.cut(
                    duration_vals,
                    bins=q,
                    labels=labels,
                    include_lowest=True,
                ).astype("string").fillna("unknown")

                for idx, cat_col in enumerate([c for c in interaction_cats if c in out.columns][:2], start=1):
                    out[f"DurationBandCombo_{idx}"] = (
                        out["DurationBand"].astype("string")
                        + "|"
                        + out[cat_col].astype("string")
                    )

    # Dynamic interaction between a primary amount and its per-duration counterpart.
    if per_duration_pairs:
        src_col, per_col = per_duration_pairs[0]
        if src_col in out.columns and per_col in out.columns:
            out["PrimaryAmountMinusPerDuration"] = (
                pd.to_numeric(out[src_col], errors="coerce").astype("float32")
                - pd.to_numeric(out[per_col], errors="coerce").astype("float32")
            ).astype("float32")

    # Derive stable temporal features from dynamically detected time-like columns.
    if time_col and time_col in out.columns:
        time_idx = _derive_time_index(out[time_col])
        if float(time_idx.notna().mean()) >= 0.30:
            out["TimeIdx"] = time_idx.astype("float32")
            radians = (2.0 * np.pi * (time_idx - 1.0) / 12.0).astype("float32")
            out["TimeSin"] = np.sin(radians).astype(np.float32)
            out["TimeCos"] = np.cos(radians).astype(np.float32)

            if duration_col and duration_col in out.columns:
                duration_vals = pd.to_numeric(out[duration_col], errors="coerce").astype("float32")
                fill_time = np.float32(time_idx.dropna().median()) if time_idx.notna().any() else np.float32(6.0)
                out["DurationTimePhase"] = (
                    duration_vals / (time_idx.fillna(fill_time).astype("float32") + np.float32(1.0))
                ).astype(np.float32)

    return out

def handle_missing_values(df: pd.DataFrame):
    """
    Fill in missing values: median for numeric, 'unknown' for categorical
    """

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if "unknown" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["unknown"])
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna("unknown")

    return df

def build_missing_fill_plan(df: pd.DataFrame):
    """
    Build deterministic fill values from the training frame for chunked inference.
    """
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    if num_cols:
        num_fill = (
            df[num_cols]
            .median(numeric_only=True)
            .to_dict()
        )
    else:
        num_fill = {}

    return {"num_fill": num_fill, "cat_cols": cat_cols, "sentinel_num_cols": [], "sentinel_cat_cols": []}


def build_missing_fill_plan_with_sentinels(
    df: pd.DataFrame,
    sentinel_cols: list[str] | None = None,
    numeric_sentinel: float = -999.0,
):
    """
    Build fill plan with explicit sentinel handling for drifted missingness columns.
    """
    plan = build_missing_fill_plan(df)
    if not sentinel_cols:
        return plan

    sentinel_num_cols: list[str] = []
    sentinel_cat_cols: list[str] = []
    for col in sentinel_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            sentinel_num_cols.append(col)
            if col in plan["num_fill"]:
                del plan["num_fill"][col]
        else:
            sentinel_cat_cols.append(col)
            if col in plan["cat_cols"]:
                plan["cat_cols"].remove(col)

    plan["sentinel_num_cols"] = sentinel_num_cols
    plan["sentinel_cat_cols"] = sentinel_cat_cols
    plan["numeric_sentinel"] = float(numeric_sentinel)
    return plan

def apply_missing_fill_plan(df: pd.DataFrame, fill_plan: dict):
    """
    Apply precomputed train-derived missing-value fills to a dataframe/chunk.
    """
    num_fill = fill_plan.get("num_fill", {})
    cat_cols = fill_plan.get("cat_cols", [])
    sentinel_num_cols = fill_plan.get("sentinel_num_cols", [])
    sentinel_cat_cols = fill_plan.get("sentinel_cat_cols", [])
    numeric_sentinel = float(fill_plan.get("numeric_sentinel", -999.0))

    for col in sentinel_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(numeric_sentinel)

    for col in sentinel_cat_cols:
        if col not in df.columns:
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if "missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["missing"])
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna("missing")

    for col, val in num_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col in cat_cols:
        if col not in df.columns:
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if "unknown" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["unknown"])
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna("unknown")
    return df

def fit_ordinal_encoder(X_train: pd.DataFrame):
    """
    Fit an ordinal encoder on train categorical columns.
    Optimized to output 32-bit integers instead of 64-bit floats.
    """
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for col in cat_cols:
        if not isinstance(X_train[col].dtype, pd.CategoricalDtype):
            try:
                X_train[col] = X_train[col].astype("category")
            except Exception:
                pass
    
    # CRITICAL: Set dtype to np.int32. 
    # LightGBM handles integers natively and faster than floats.
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-2,
        dtype=np.int32 
    )
    
    if cat_cols:
        fit_frame = X_train.copy()
        for col in cat_cols:
            if col not in fit_frame.columns:
                fit_frame[col] = "unknown"
            col_s = fit_frame[col].astype("string").fillna("unknown")
            fit_frame[col] = col_s.astype(str)
        encoder.fit(fit_frame[cat_cols])
        
    return encoder, cat_cols

def apply_ordinal_encoder(df: pd.DataFrame, encoder: OrdinalEncoder, cat_cols: list[str]):
    """
    Apply a pre-fitted ordinal encoder to any dataframe chunk.
    """
    out = df.copy()
    if not cat_cols:
        return out

    for col in cat_cols:
        if col not in out.columns:
            out[col] = "unknown"

    transform_frame = out.copy()
    for col in cat_cols:
        col_s = transform_frame[col].astype("string").fillna("unknown")
        transform_frame[col] = col_s.astype(str)

    out[cat_cols] = encoder.transform(transform_frame[cat_cols])
    return out

def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame, return_encoder: bool = False):
    """
    Converts text to number using Ordinal Encoding 
    """
    for frame in (X_train, X_test):
        cat_like_cols = frame.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        for col in cat_like_cols:
            if not isinstance(frame[col].dtype, pd.CategoricalDtype):
                try:
                    frame[col] = frame[col].astype("category")
                except Exception:
                    pass

    encoder, cat_cols = fit_ordinal_encoder(X_train)
    X_train_enc = apply_ordinal_encoder(X_train, encoder, cat_cols)
    X_test_enc = apply_ordinal_encoder(X_test, encoder, cat_cols)

    if return_encoder:
        return X_train_enc, X_test_enc, encoder, cat_cols
    return X_train_enc, X_test_enc

def convert_string_columns_to_category(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_unique: int = 5000,
    max_unique_ratio: float = 0.20,
    sample_rows: int = 100000
):
    """
    Convert selected string columns to categorical dtype to lower memory and speed groupby/value_counts.
    We skip extremely high-cardinality columns where category dtype may not help.
    """
    object_cols = [
        c for c in X_train.select_dtypes(include=['object', 'category', 'string']).columns
        if c in X_test.columns
    ]

    converted = []

    for col in object_cols:
        tr_non_na = X_train[col].dropna()
        te_non_na = X_test[col].dropna()

        tr_sample = tr_non_na.iloc[: min(len(tr_non_na), sample_rows)]
        te_sample = te_non_na.iloc[: min(len(te_non_na), sample_rows)]

        tr_unique = tr_sample.nunique(dropna=True)
        te_unique = te_sample.nunique(dropna=True)
        combined_unique = max(tr_unique, te_unique)
        sample_total = len(tr_sample) + len(te_sample)
        unique_ratio = combined_unique / max(1, sample_total)

        if combined_unique <= max_unique or unique_ratio <= max_unique_ratio:
            try:
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")
                converted.append(col)
            except Exception:
                continue

    return X_train, X_test, converted

def engineer_features(df: pd.DataFrame):
    """
    Creates robust engineered features using dtype-driven primitives.
    """
    pl_df = _to_polars(df)

    # 1. Binary-positive density from text categorical columns.
    existing_cols = _detect_binary_positive_text_columns(df)
    exprs = []

    if existing_cols:
        binary_exprs = [
            (
                pl.col(c)
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .eq("yes")
                .cast(pl.Int8)
            )
            for c in existing_cols
        ]
        exprs.append(
            pl.sum_horizontal(binary_exprs)
            .cast(pl.Int8)
            .alias('Total_Binary_Positive_Count')
        )
    else:
        exprs.append(pl.lit(0, dtype=pl.Int8).alias('Total_Binary_Positive_Count'))

    # 2. Geospatial Gridding (Round Lat/Lon to group neighbors together)
    lat_col, lon_col = _detect_lat_lon_columns(df)
    if lat_col and lon_col:
        exprs.append(pl.col(lat_col).round(1).alias('Lat_Grid'))
        exprs.append(pl.col(lon_col).round(1).alias('Lon_Grid'))

    out = pl_df.with_columns(exprs).to_pandas()
    return _apply_percentile_rank_transformation(out)

def add_missingness_indicators(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    min_shift: float = 0.05,
    min_missing_rate: float = 0.01,
    max_features: int = 20
):
    """
    Add binary missingness flags for columns whose NA rate shifts between train/test.
    Optimized with np.int8 to minimize memory footprint.
    """
    shared_cols = X_train.columns.intersection(X_test.columns)
    if shared_cols.empty or max_features <= 0:
        return X_train, X_test, []

    # REFACTORED: fully vectorized NA-rate and shift computation across the full column set.
    train_na = X_train[shared_cols].isna().mean(axis=0)
    test_na = X_test[shared_cols].isna().mean(axis=0)
    shift = (train_na - test_na).abs()
    max_missing = pd.concat([train_na, test_na], axis=1).max(axis=1)

    candidate_shift = shift[(shift >= min_shift) & (max_missing >= min_missing_rate)]
    top_cols = candidate_shift.sort_values(ascending=False, kind="mergesort").head(max_features).index.tolist()
    if not top_cols:
        return X_train, X_test, []

    indicator_cols = [f"{col}__was_missing" for col in top_cols]

    train_missing = X_train[top_cols].isna().astype(np.int8)
    train_missing.columns = indicator_cols
    X_train[indicator_cols] = train_missing

    test_missing = X_test[top_cols].isna().astype(np.int8)
    test_missing.columns = indicator_cols
    X_test[indicator_cols] = test_missing

    return X_train, X_test, indicator_cols
