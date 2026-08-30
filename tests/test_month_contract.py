from __future__ import annotations

import pandas as pd
import pytest

from common.contracts import parse_competition_month_value, sorted_month_values


def test_every_competition_month_abbreviation() -> None:
    abbreviations = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    raw = pd.Series([f"25-{month}" for month in abbreviations], name="Month")
    original = raw.copy()

    parsed = [parse_competition_month_value(value) for value in raw]
    assert parsed == list(range(2025 * 12, 2025 * 12 + 12))
    pd.testing.assert_series_equal(raw, original)


def test_cross_year_and_shuffled_chronology() -> None:
    shuffled = pd.Series(["26-Feb", "25-Dec", "25-Nov", "26-Jan", "25-Dec"])
    assert sorted_month_values(shuffled) == ["25-Nov", "25-Dec", "26-Jan", "26-Feb"]

    order = [
        parse_competition_month_value(value)
        for value in ["25-Nov", "25-Dec", "26-Jan", "26-Feb"]
    ]
    assert [right - left for left, right in zip(order, order[1:])] == [1, 1, 1]


@pytest.mark.parametrize("value", ["25-Jax", "2025-Jan", None, "Jan-25"])
def test_invalid_competition_month_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        parse_competition_month_value(value)
