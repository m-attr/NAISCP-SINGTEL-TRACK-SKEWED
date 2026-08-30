from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="session")
def public_data_paths() -> tuple[Path, Path]:
    train_path = REPO_ROOT / "train.csv"
    test_path = REPO_ROOT / "test.csv"
    if not train_path.exists() or not test_path.exists():
        pytest.skip("Public train.csv/test.csv are not available in the repository working directory.")
    return train_path, test_path


@pytest.fixture(scope="session")
def bounded_public_data(public_data_paths: tuple[Path, Path], tmp_path_factory) -> tuple[Path, Path, pd.DataFrame]:
    train_path, test_path = public_data_paths
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    pieces: list[pd.DataFrame] = []
    grouped = train.groupby(["Month", "ChurnStatus"], sort=True, observed=True)
    for group_index, (_, group) in enumerate(grouped):
        pieces.append(group.sample(n=min(80, len(group)), random_state=4200 + group_index))
    bounded_train = pd.concat(pieces).sort_index(kind="stable").reset_index(drop=True)
    bounded_test = test.sample(n=min(600, len(test)), random_state=4300).sort_index(kind="stable").reset_index(drop=True)

    data_dir = tmp_path_factory.mktemp("bounded_public_data")
    bounded_train_path = data_dir / "train.csv"
    bounded_test_path = data_dir / "test.csv"
    bounded_train.to_csv(bounded_train_path, index=False)
    bounded_test.to_csv(bounded_test_path, index=False)
    return bounded_train_path, bounded_test_path, bounded_test
