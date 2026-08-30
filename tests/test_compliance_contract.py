from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from model import lightgbm_model as model_trainer
from preprocessing.plan import build_preparation_plan, transform_test
from runtime.streaming import select_test_feature_columns


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _literal_keyword_map(call: ast.Call) -> dict[str, object]:
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg is not None}


def test_official_lightgbm_constructor_is_exact() -> None:
    tree = ast.parse((SRC_ROOT / "model" / "lightgbm_model.py").read_text(encoding="utf-8"))
    factory = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_official_lightgbm_model"
    )
    constructors = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LGBMClassifier"
    ]
    assert len(constructors) == 1
    assert constructors[0].args == []
    assert _literal_keyword_map(constructors[0]) == {
        "verbosity": -1,
        "objective": "binary",
        "is_unbalance": True,
        "random_state": 42,
        "importance_type": "gain",
    }
    assert "lightgbm==4.6.0" in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()


def test_production_source_has_one_fit_and_no_prohibited_paths() -> None:
    production_files = [
        path
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    fit_locations: list[tuple[str, int]] = []
    prohibited_text = {
        "MiniBatchKMeans": "KMeans row-selection fitting",
        "OrdinalEncoder": "fitted category encoder",
        "fit_predict": "fitted row-selection helper",
        "lgb.train": "raw LightGBM training API",
        "train_and_predict_raw_baseline": "raw baseline training",
        "feature_gating": "score-gated pipeline rerun",
        "subprocess.run": "pipeline subprocess rerun",
        "test_auprc": "production test-label scoring surface",
        "y_test": "production test target surface",
    }

    for path in production_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
                fit_locations.append((path.relative_to(SRC_ROOT).as_posix(), node.lineno))
        for needle, description in prohibited_text.items():
            assert needle not in source, f"Found prohibited {description} in {path.name}."

    assert len(fit_locations) == 1
    assert fit_locations[0][0] == "model/lightgbm_model.py"
    assert "evaluate_public_predictions" not in (
        SRC_ROOT / "pipeline" / "orchestrator.py"
    ).read_text(encoding="utf-8")


def test_model_fit_instrumentation_and_constructor(monkeypatch) -> None:
    calls: dict[str, object] = {"fit_count": 0}

    class RecordingClassifier:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs

        def fit(self, X, y, **kwargs):
            calls["fit_count"] = int(calls["fit_count"]) + 1
            calls["fit_shape"] = np.asarray(X).shape
            return self

        def predict_proba(self, X):
            n_rows = len(X)
            scores = np.linspace(0.2, 0.8, n_rows, dtype=np.float64)
            return np.column_stack([1.0 - scores, scores])

    monkeypatch.setattr(model_trainer, "LGBMClassifier", RecordingClassifier)
    train = pd.DataFrame({"numeric": np.arange(40), "category": [0, 1] * 20})
    target = pd.Series([0, 1] * 20)
    model_trainer.reset_production_model_fit_count()
    model, train_auprc = model_trainer.fit_once(
        train,
        target.to_numpy(dtype=np.int8),
        categorical_features=["category"],
    )
    predictions = model_trainer.predict_in_chunks(model, train.iloc[:2], chunk_size=1)

    assert calls["constructor"] == {
        "verbosity": -1,
        "objective": "binary",
        "is_unbalance": True,
        "random_state": 42,
        "importance_type": "gain",
    }
    assert calls["fit_count"] == 1
    assert model_trainer.get_production_model_fit_count() == 1
    assert len(predictions) == 2
    assert 0.0 <= train_auprc <= 1.0


def test_test_target_drop_never_reads_or_maps_values(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "CustomerID": [101, 102, 103, 104],
            "Month": ["25-Nov", "25-Nov", "25-Dec", "25-Dec"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    variants = [
        features.assign(ChurnStatus=["No", "Yes", "No", "Yes"]),
        features.assign(ChurnStatus=0),
        features.assign(ChurnStatus=1),
        features.assign(ChurnStatus=["Yes", "No", "Yes", "No"]),
        features.copy(),
    ]
    for index, variant in enumerate(variants):
        path = tmp_path / f"test_variant_{index}.csv"
        variant.to_csv(path, index=False)
        selected = select_test_feature_columns(path)
        assert "ChurnStatus" not in selected
        pd.testing.assert_frame_equal(pd.read_csv(path, usecols=selected), features)


def test_category_mappings_are_deterministic_and_reused() -> None:
    train = pd.DataFrame(
        {
            "CustomerID": np.arange(40),
            "Month": ["25-Nov", "25-Dec"] * 20,
            "ChurnStatus": ["No", "Yes"] * 20,
            "category": ["b", "a", None, "b"] * 10,
            "numeric": np.arange(40, dtype=np.float64),
        }
    )
    test = pd.DataFrame(
        {
            "CustomerID": [101, 102, 103],
            "Month": ["25-Dec"] * 3,
            "category": ["a", "unknown", None],
            "numeric": [5.0, 6.0, 7.0],
        }
    )
    shuffled = train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    plan_a = build_preparation_plan(
        train,
        test,
        model_trainer.target_to_array(train["ChurnStatus"]),
    )
    plan_b = build_preparation_plan(
        shuffled,
        test,
        model_trainer.target_to_array(shuffled["ChurnStatus"]),
    )
    assert plan_a.category_maps == plan_b.category_maps
    assert plan_a.categorical_features == plan_b.categorical_features == ["category"]

    transformed = transform_test(test, plan_a)
    mapping = plan_a.category_maps["category"]
    assert transformed["category"].tolist() == [mapping["a"], -1, mapping["__missing__"]]
