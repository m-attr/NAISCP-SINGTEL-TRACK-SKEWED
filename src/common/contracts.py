from __future__ import annotations

import re
from typing import Any

import pandas as pd

ID_COLUMN = "CustomerID"
TIME_COLUMN = "Month"
TARGET_COLUMN = "ChurnStatus"
MISSING_CATEGORY = "__missing__"
POOLED_MONTH = "__all__"

_MONTH_NUMBER = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
_MONTH_PATTERN = re.compile(r"^(?P<year>\d{2})-(?P<month>[A-Za-z]{3})$")


def canonicalize_category(series: pd.Series) -> pd.Series:
    """Normalize category text without relying on any non-contract feature name."""
    return (
        series.astype("string")
        .fillna(MISSING_CATEGORY)
        .str.strip()
        .str.lower()
        .replace("", MISSING_CATEGORY)
    )


def parse_competition_month_value(value: Any) -> int:
    """Convert the competition YY-MMM value to one cross-year sortable integer."""
    match = _MONTH_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(
            f"Invalid Month value {value!r}; expected YY-MMM, for example 25-Jan."
        )
    month_text = match.group("month").title()
    if month_text not in _MONTH_NUMBER:
        raise ValueError(f"Invalid Month abbreviation in {value!r}.")
    year = 2000 + int(match.group("year"))
    return year * 12 + _MONTH_NUMBER[month_text] - 1


def sorted_month_values(series: pd.Series) -> list[str]:
    values = [str(value).strip() for value in series.dropna().unique().tolist()]
    return sorted(values, key=parse_competition_month_value)
