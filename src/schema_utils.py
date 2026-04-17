import re
from typing import Iterable

import numpy as np
import pandas as pd


def name_tokens(name: str) -> set[str]:
    raw = str(name).strip()
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    raw = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", raw)
    return {t for t in re.split(r"[^a-z0-9]+", raw.lower()) if t}


def is_identifier_name(name: str) -> bool:
    tokens = name_tokens(name)
    if not tokens:
        return False
    if "id" in tokens:
        return True
    return any(tok in {"uuid", "guid", "identifier", "key", "account"} for tok in tokens)


def is_time_like_name(name: str) -> bool:
    tokens = name_tokens(name)
    return any(tok in {"month", "date", "time", "period", "week", "quarter", "year", "timestamp"} for tok in tokens)


def _binary_like_score(series: pd.Series) -> float:
    non_na = series.dropna()
    if non_na.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(non_na):
        uniq = pd.Series(non_na).nunique(dropna=True)
        return 1.0 if uniq <= 2 else 0.0

    lowered = non_na.astype("string").str.strip().str.lower()
    uniq_vals = [u for u in lowered.unique().tolist() if u is not pd.NA]
    if len(uniq_vals) != 2:
        return 0.0
    known_binary = {
        "0", "1", "yes", "no", "true", "false", "y", "n", "churn", "stay", "positive", "negative"
    }
    hit = sum(1 for v in uniq_vals if str(v) in known_binary)
    return 0.5 + 0.25 * hit


def detect_identifier_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    candidates: list[tuple[str, float]] = []
    for col in df.columns:
        s = df[col]
        non_na = s.dropna()
        if len(non_na) < 100:
            continue

        tokens = name_tokens(col)
        score = 0.0
        if is_identifier_name(col):
            score += 3.0
        if is_time_like_name(col):
            score -= 2.0

        uniq_ratio = float(non_na.nunique(dropna=True)) / max(1, len(non_na))
        if uniq_ratio >= 0.95:
            score += 1.5
        if float(non_na.notna().mean()) >= 0.95:
            score += 0.5

        if pd.api.types.is_numeric_dtype(s):
            arr = pd.to_numeric(non_na, errors="coerce").to_numpy(dtype=np.float64, copy=False)
            int_like = float(np.isclose(np.mod(arr, 1.0), 0.0, atol=1e-8).mean()) if len(arr) else 0.0
            if int_like >= 0.95:
                score += 0.5
            if not is_identifier_name(col):
                score -= 0.75

        if score >= 3.0:
            candidates.append((col, score))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_time_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    text_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    best_col = None
    best_score = -1.0
    month_pat = re.compile(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", flags=re.IGNORECASE)

    for col in text_cols:
        s = df[col].dropna().astype("string")
        if s.empty:
            continue
        s = s.iloc[: min(len(s), 4000)]

        month_ratio = float(s.str.contains(month_pat, na=False, regex=True).mean())
        try:
            dt_parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        except Exception:
            dt_parsed = pd.to_datetime(s, errors="coerce")
        dt_ratio = float(dt_parsed.notna().mean())

        score = max(month_ratio, dt_ratio)
        if is_time_like_name(col):
            score += 0.20

        if score > best_score:
            best_score = score
            best_col = col

    if best_col is not None and best_score >= 0.45:
        return best_col

    # Fallback for pre-engineered numeric time indices.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if not is_time_like_name(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 100:
            continue
        nunique = int(s.nunique(dropna=True))
        if 3 <= nunique <= 128:
            return col
    return None


def detect_target_column(df: pd.DataFrame, preferred: Iterable[str] | None = None) -> str | None:
    if df.empty:
        return None

    if preferred:
        for cand in preferred:
            if cand in df.columns:
                return cand

    best_col = None
    best_score = -1.0

    for col in df.columns:
        if is_identifier_name(col) or is_time_like_name(col):
            continue

        s = df[col]
        non_na = s.dropna()
        if len(non_na) < 100:
            continue

        nunique = int(non_na.nunique(dropna=True))
        if nunique < 2 or nunique > 12:
            continue

        tokens = name_tokens(col)
        score = 0.0
        if any(t in {"target", "label", "outcome", "response", "class"} for t in tokens):
            score += 3.0
        if any(t in {"churn", "default", "fraud", "risk"} for t in tokens):
            score += 2.0
        if any(t in {"status", "flag"} for t in tokens):
            score += 0.75
        score += _binary_like_score(non_na)

        # Penalize columns that look like ordinary features despite being low-card.
        if any(t in {"type", "plan", "contract", "offer", "state", "city", "country"} for t in tokens):
            score -= 1.0

        if score > best_score:
            best_score = score
            best_col = col

    return best_col if best_score >= 1.25 else None
