from __future__ import annotations

import numpy as np
import pandas as pd

from model.lightgbm_model import target_to_array
from preprocessing.plan import build_preparation_plan, transform_test


def _transform_in_chunks(frame: pd.DataFrame, plan, chunk_count: int) -> pd.DataFrame:
    boundaries = np.linspace(0, len(frame), num=chunk_count + 1, dtype=int)
    parts = [
        transform_test(frame.iloc[boundaries[i] : boundaries[i + 1]].copy(), plan)
        for i in range(chunk_count)
        if boundaries[i] < boundaries[i + 1]
    ]
    return pd.concat(parts, axis=0, ignore_index=True)


def test_frozen_preparation_plan_is_chunk_invariant(bounded_public_data) -> None:
    train_path, test_path, _ = bounded_public_data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path).drop(columns=["ChurnStatus"], errors="ignore")
    target = target_to_array(train["ChurnStatus"])
    plan = build_preparation_plan(train, test, target)

    whole = transform_test(test, plan).reset_index(drop=True)
    for chunk_count in (1, 10, 100):
        chunked = _transform_in_chunks(test, plan, chunk_count)
        pd.testing.assert_frame_equal(whole, chunked, check_exact=True)
