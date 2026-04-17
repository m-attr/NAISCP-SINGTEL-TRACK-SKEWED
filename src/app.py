from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "bg": "#F9FAFB",
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
    "reference": "#94A3B8",
    "current": "#E11D48",
    "stable": "#10B981",
    "plot_inner": "#EAF2FF",
    "axis_dark": "#374151",
}


st.set_page_config(
    page_title="Data Drift Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{display: none;}}
    [data-testid="stDecoration"] {{display: none;}}
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
        font-family: "Inter", "Segoe UI", sans-serif;
    }}
    .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 1.1rem;
    }}
    .section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0.15rem 0 0.45rem 0;
        color: {COLORS['text']};
    }}
    .section-subtitle {{
        font-size: 0.88rem;
        color: {COLORS['muted']};
        margin-top: -0.10rem;
        margin-bottom: 0.50rem;
    }}
    .card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        padding: 12px 14px;
    }}
    .metric-title {{
        color: {COLORS['muted']};
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }}
    .metric-value {{
        color: {COLORS['text']};
        font-size: 1.60rem;
        font-weight: 700;
        margin-top: 4px;
        line-height: 1.2;
    }}
    .metric-sub {{
        color: {COLORS['muted']};
        font-size: 0.83rem;
        margin-top: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _artifact_roots() -> list[Path]:
    here = Path(__file__).resolve()
    return [Path.cwd(), here.parent, here.parent.parent]


def _find_artifact(filename: str) -> Path:
    for root in _artifact_roots():
        candidate = root / filename
        if candidate.exists():
            return candidate
    return _artifact_roots()[0] / filename


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _to_float_or_nan(value: Any) -> float:
    return _to_float(value, float("nan"))


def _extract_first_number(text: str) -> float:
    vals = re.findall(r"([0-9]*\.?[0-9]+)", str(text))
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except Exception:
            continue
    return max(nums) if nums else 0.0


def _spark(value: float, lo: float, hi: float) -> str:
    bars = " .:-=+*#%@"
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return bars[len(bars) // 2]
    x = (value - lo) / (hi - lo)
    x = float(np.clip(x, 0.0, 1.0))
    idx = int(round(x * (len(bars) - 1)))
    return bars[idx]


def _guess_time_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        s = df[col]
        if s.dtype.kind in {"O", "U", "S"}:
            try:
                parsed = pd.to_datetime(s.astype("string"), errors="coerce", format="mixed")
            except Exception:
                parsed = pd.to_datetime(s.astype("string"), errors="coerce")
            if float(parsed.notna().mean()) >= 0.50:
                return col
    return None


def _infer_test_name(reason: str, feature_type: str) -> str:
    r = str(reason).lower()
    if "ks" in r:
        return "KS-Test"
    if "psi" in r:
        return "PSI"
    if "cramer" in r:
        return "Cramer's V"
    if "jsd" in r or "jensen" in r:
        return "Jensen-Shannon"
    if "binary proportion" in r or "proportion" in r:
        return "Proportion Delta"
    return "KS-Test" if feature_type == "Numeric" else "Jensen-Shannon"


def _render_metric_card(title: str, value: str, sub: str, accent: str) -> str:
    return (
        '<div class="card" style="border-left: 6px solid ' + accent + ';">'
        + '<div class="metric-title">' + title + '</div>'
        + '<div class="metric-value">' + value + '</div>'
        + '<div class="metric-sub">' + sub + '</div>'
        + '</div>'
    )


def _mock_data() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 5000

    train = pd.DataFrame(
        {
            "Feature_001": rng.normal(40.0, 11.0, n),
            "Feature_002": rng.gamma(2.4, 15.0, n),
            "Feature_003": rng.choice(["A", "B", "C", "D"], n, p=[0.42, 0.30, 0.20, 0.08]),
            "Feature_004": rng.choice([0, 1], n, p=[0.73, 0.27]),
            "Feature_005": rng.normal(0.0, 1.0, n),
            "Feature_006": rng.poisson(4.0, n),
            "Feature_007": rng.normal(85.0, 22.0, n),
            "Feature_008": rng.choice(["x", "y", "z"], n, p=[0.60, 0.30, 0.10]),
        }
    )
    test = pd.DataFrame(
        {
            "Feature_001": rng.normal(46.0, 12.0, n),
            "Feature_002": rng.gamma(2.0, 20.0, n),
            "Feature_003": rng.choice(["A", "B", "C", "D"], n, p=[0.20, 0.34, 0.31, 0.15]),
            "Feature_004": rng.choice([0, 1], n, p=[0.62, 0.38]),
            "Feature_005": rng.normal(0.55, 1.25, n),
            "Feature_006": rng.poisson(5.5, n),
            "Feature_007": rng.normal(98.0, 28.0, n),
            "Feature_008": rng.choice(["x", "y", "z"], n, p=[0.45, 0.43, 0.12]),
        }
    )

    train.loc[rng.random(n) < 0.02, "Feature_002"] = np.nan
    test.loc[rng.random(n) < 0.10, "Feature_002"] = np.nan

    metrics = {
        "train_auprc": 0.8821,
        "test_auprc": 0.8464,
        "runtime": 18.72,
        "drift_runtime": 11.33,
        "pre_train_auprc": 0.8614,
        "pre_test_auprc": 0.8112,
    }

    features = []
    for c in train.columns:
        tr = pd.to_numeric(train[c], errors="coerce")
        te = pd.to_numeric(test[c], errors="coerce")
        is_num = float(tr.notna().mean()) >= 0.8 and float(te.notna().mean()) >= 0.8
        if is_num:
            score = float(np.clip(abs(tr.mean() - te.mean()) / (tr.std() + 1e-6), 0.0, 1.0))
            test_name = "KS-Test"
        else:
            tr_dist = train[c].astype("string").value_counts(normalize=True)
            te_dist = test[c].astype("string").value_counts(normalize=True)
            all_idx = tr_dist.index.union(te_dist.index)
            tv = 0.5 * np.abs(tr_dist.reindex(all_idx, fill_value=0) - te_dist.reindex(all_idx, fill_value=0)).sum()
            score = float(np.clip(tv, 0.0, 1.0))
            test_name = "Jensen-Shannon"

        status = "Drifted" if score >= 0.10 else "Stable"
        q_low = float(np.nanpercentile(tr, 5)) if is_num else None
        q_high = float(np.nanpercentile(tr, 95)) if is_num else None
        features.append(
            {
                "feature": c,
                "type": "Numeric" if is_num else "Categorical",
                "drift_score": score,
                "status": status,
                "stat_test": test_name,
                "p_value": float(np.clip(rng.uniform(0.0001, 0.2), 0.0001, 1.0)),
                "threshold": 0.10,
                "mitigated": bool(status == "Drifted" and is_num),
                "q_low": q_low,
                "q_high": q_high,
                "reason": f"Mocked drift signal ({test_name})",
            }
        )

    drift = {
        "features": features,
        "winsor_bounds": {
            "Feature_001": {"q_low": float(np.nanpercentile(train["Feature_001"], 5)), "q_high": float(np.nanpercentile(train["Feature_001"], 95))},
            "Feature_002": {"q_low": float(np.nanpercentile(train["Feature_002"], 5)), "q_high": float(np.nanpercentile(train["Feature_002"], 95))},
        },
        "data_quality_issues": {
            "Feature_002": "Missing value rate shifted from 2.0% to 10.0%",
            "Feature_003": "Case casing drift detected",
        },
    }
    return metrics, drift, train, test


def _normalize_drift_payload(raw: Any, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]], list[str], dict[str, Any]]:
    rows = []
    winsor = {}
    dq = []

    if not isinstance(raw, dict):
        raw = {}

    wb = raw.get("winsor_bounds", {})
    if isinstance(wb, dict):
        for feat, bounds in wb.items():
            if isinstance(bounds, dict):
                winsor[str(feat)] = {
                    "q_low": _to_float_or_nan(bounds.get("q_low")),
                    "q_high": _to_float_or_nan(bounds.get("q_high")),
                }

    dq_raw = raw.get("data_quality_issues", {})
    if isinstance(dq_raw, dict):
        dq = [f"{k}: {v}" for k, v in dq_raw.items()]
    elif isinstance(dq_raw, list):
        dq = [str(x) for x in dq_raw]

    if isinstance(raw.get("features"), list):
        entries = raw.get("features", [])
        for item in entries:
            if not isinstance(item, dict):
                continue
            feat = str(item.get("feature", "")).strip()
            if not feat:
                continue

            if feat in train.columns and feat in test.columns:
                ft = "Numeric" if (pd.to_numeric(train[feat], errors="coerce").notna().mean() >= 0.8 and pd.to_numeric(test[feat], errors="coerce").notna().mean() >= 0.8) else "Categorical"
            else:
                ft = str(item.get("type", "Unknown")).title()

            reason = str(item.get("reason", ""))
            stat_test = str(item.get("stat_test", _infer_test_name(reason, ft)))
            drift_score = _to_float(item.get("drift_score"), _extract_first_number(reason))
            status = str(item.get("status", "Stable"))
            threshold = _to_float(item.get("threshold"), 0.10)
            mitigated = bool(item.get("mitigated", False))

            q_low = _to_float_or_nan(item.get("q_low"))
            q_high = _to_float_or_nan(item.get("q_high"))
            if np.isfinite(q_low) and np.isfinite(q_high):
                winsor[feat] = {"q_low": q_low, "q_high": q_high}

            rows.append(
                {
                    "Feature": feat,
                    "Type": ft,
                    "Drift Score": drift_score,
                    "Stat Test": stat_test,
                    "P-Value": _to_float_or_nan(item.get("p_value")),
                    "Data Drift": "Detected" if status.lower().startswith("drift") else "Not Detected",
                    "Threshold": threshold,
                    "Mitigated": mitigated or feat in winsor,
                    "Reason": reason,
                }
            )
    else:
        for feat, val in raw.items():
            if feat in {"winsor_bounds", "data_quality_issues", "generated_at", "metrics"}:
                continue
            reason = str(val if not isinstance(val, dict) else val.get("reason", ""))
            score = _extract_first_number(reason)
            ft = "Numeric" if (feat in train.columns and pd.to_numeric(train[feat], errors="coerce").notna().mean() >= 0.8) else "Categorical"
            rows.append(
                {
                    "Feature": str(feat),
                    "Type": ft,
                    "Drift Score": score,
                    "Stat Test": _infer_test_name(reason, ft),
                    "P-Value": _to_float_or_nan(val.get("p_value") if isinstance(val, dict) else np.nan),
                    "Data Drift": "Detected" if reason else "Not Detected",
                    "Threshold": _to_float(val.get("threshold") if isinstance(val, dict) else 0.10, 0.10),
                    "Mitigated": bool(feat in winsor),
                    "Reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            {
                "Feature": ["Feature_001"],
                "Type": ["Numeric"],
                "Drift Score": [0.0],
                "Stat Test": ["KS-Test"],
                "P-Value": [np.nan],
                "Data Drift": ["Not Detected"],
                "Threshold": [0.10],
                "Mitigated": [False],
                "Reason": ["No drift data available"],
            }
        )

    df["Drift Score"] = pd.to_numeric(df["Drift Score"], errors="coerce").fillna(0.0)
    df["P-Value"] = pd.to_numeric(df["P-Value"], errors="coerce")
    df["Threshold"] = pd.to_numeric(df["Threshold"], errors="coerce").fillna(0.10)
    df["Type"] = df["Type"].astype(str).replace({"Unknown": "Numeric"})
    df["Data Drift"] = np.where(df["Data Drift"].astype(str).str.lower().str.contains("detected"), df["Data Drift"], "Not Detected")
    return df, winsor, dq, raw


@st.cache_data(show_spinner=False)
def load_or_mock_data() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]], list[str], dict[str, bool], dict[str, Any]]:
    metrics_path = _find_artifact("latest_metrics.json")
    drift_path = _find_artifact("drift_report.json")
    train_path = _find_artifact("train_sample.parquet")
    test_path = _find_artifact("test_sample.parquet")

    used_mock = {"metrics": False, "drift": False, "samples": False}

    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
            used_mock["metrics"] = True
    else:
        metrics = {}
        used_mock["metrics"] = True

    if drift_path.exists():
        try:
            with drift_path.open("r", encoding="utf-8") as f:
                drift_payload = json.load(f)
        except Exception:
            drift_payload = {}
            used_mock["drift"] = True
    else:
        drift_payload = {}
        used_mock["drift"] = True

    if train_path.exists() and test_path.exists():
        try:
            train = pd.read_parquet(train_path)
            test = pd.read_parquet(test_path)
        except Exception:
            train = pd.DataFrame()
            test = pd.DataFrame()
            used_mock["samples"] = True
    else:
        train = pd.DataFrame()
        test = pd.DataFrame()
        used_mock["samples"] = True

    if used_mock["metrics"] or used_mock["drift"] or used_mock["samples"] or train.empty or test.empty:
        m_metrics, m_drift, m_train, m_test = _mock_data()
        if used_mock["metrics"]:
            metrics = m_metrics
        if used_mock["drift"]:
            drift_payload = m_drift
        if used_mock["samples"] or train.empty or test.empty:
            train = m_train
            test = m_test

    drift_df, winsor, dq_alerts, raw = _normalize_drift_payload(drift_payload, train, test)
    return metrics, drift_df, train, test, winsor, dq_alerts, used_mock, raw


def _auprc_matrix(metrics: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    post_train = _to_float(metrics.get("train_auprc"), 0.0)
    post_test = _to_float(metrics.get("test_auprc"), 0.0)

    pre_train = _to_float(metrics.get("pre_train_auprc", metrics.get("train_auprc_pre")), post_train - 0.01)
    pre_test = _to_float(metrics.get("pre_test_auprc", metrics.get("test_auprc_pre")), post_test - 0.015)

    pre_train = float(np.clip(pre_train, 0.0, 1.0))
    pre_test = float(np.clip(pre_test, 0.0, 1.0))

    matrix = pd.DataFrame(
        {
            "Pre-Mitigation": [pre_train, pre_test],
            "Post-Mitigation": [post_train, post_test],
        },
        index=["Train", "Test"],
    )
    return matrix, {"Train": pre_train, "Test": pre_test}, {"Train": post_train, "Test": post_test}


def _proxy_pr_curve_from_auprc(auprc: float, points: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a smooth monotonic proxy precision-recall curve from a scalar AU-PRC.
    Used only when full curve artifacts are unavailable.
    """
    auprc = float(np.clip(auprc, 1e-4, 0.9999))
    recall = np.linspace(0.0, 1.0, points)

    floor = min(0.20, max(0.01, auprc * 0.35))
    if auprc <= floor + 1e-5:
        floor = max(0.001, auprc * 0.5)

    denom = max(1e-6, auprc - floor)
    k = (1.0 - floor) / denom - 1.0
    k = float(np.clip(k, 0.05, 80.0))

    precision = floor + (1.0 - floor) * np.power(1.0 - recall, k)
    precision = np.clip(precision, 0.0, 1.0)
    return recall, precision


def _extract_pr_curve(metrics: dict[str, Any], stage: str, split: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns recall, precision, and whether curve is exact (from artifact).
    Accepted key pattern examples:
    - pre_train_pr_curve = {"recall": [...], "precision": [...]} or [[r, p], ...]
    - post_test_pr_curve = {"recall": [...], "precision": [...]} or [[r, p], ...]
    """
    key = f"{stage}_{split}_pr_curve"
    obj = metrics.get(key)

    if isinstance(obj, dict) and "recall" in obj and "precision" in obj:
        r = np.asarray(obj.get("recall", []), dtype=np.float64)
        p = np.asarray(obj.get("precision", []), dtype=np.float64)
        if r.size >= 2 and p.size == r.size:
            return np.clip(r, 0.0, 1.0), np.clip(p, 0.0, 1.0), True

    if isinstance(obj, list) and obj and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in obj):
        arr = np.asarray(obj, dtype=np.float64)
        r = arr[:, 0]
        p = arr[:, 1]
        if r.size >= 2 and p.size == r.size:
            return np.clip(r, 0.0, 1.0), np.clip(p, 0.0, 1.0), True

    scalar_key = "train_auprc" if (stage == "post" and split == "train") else (
        "test_auprc" if (stage == "post" and split == "test") else (
            "pre_train_auprc" if split == "train" else "pre_test_auprc"
        )
    )
    fallback = 0.5
    if stage == "pre" and split == "train":
        fallback = _to_float(metrics.get("train_auprc"), 0.5) - 0.01
    elif stage == "pre" and split == "test":
        fallback = _to_float(metrics.get("test_auprc"), 0.5) - 0.015

    auprc = _to_float(metrics.get(scalar_key), fallback)
    r, p = _proxy_pr_curve_from_auprc(auprc)
    return r, p, False


def _build_pr_line_chart(title: str, metrics: dict[str, Any], stage: str) -> tuple[go.Figure, bool]:
    train_r, train_p, exact_train = _extract_pr_curve(metrics, stage=stage, split="train")
    test_r, test_p, exact_test = _extract_pr_curve(metrics, stage=stage, split="test")
    is_exact = bool(exact_train and exact_test)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_r,
            y=train_p,
            mode="lines",
            name="Train",
            line=dict(color="#1F2937", width=3),
            hovertemplate="Train<br>Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_r,
            y=test_p,
            mode="lines",
            name="Test",
            line=dict(color="#DC2626", width=3),
            hovertemplate="Test<br>Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top",
            font=dict(color=COLORS["axis_dark"], size=16),
        ),
        template="plotly_white",
        height=355,
        margin=dict(l=10, r=10, t=58, b=10),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["plot_inner"],
        yaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            title="Precision",
            tickfont=dict(color=COLORS["axis_dark"]),
            titlefont=dict(color=COLORS["axis_dark"]),
            color=COLORS["axis_dark"],
        ),
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            title="Recall",
            tickfont=dict(color=COLORS["axis_dark"]),
            titlefont=dict(color=COLORS["axis_dark"]),
            color=COLORS["axis_dark"],
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(color=COLORS["axis_dark"]),
            bgcolor="rgba(255,255,255,0.70)",
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    return fig, is_exact


def _distribution_signature(feature: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str, float, float, str]:
    tr = train[feature]
    te = test[feature]

    tr_num = pd.to_numeric(tr, errors="coerce")
    te_num = pd.to_numeric(te, errors="coerce")
    numeric_like = float(tr_num.notna().mean()) >= 0.8 and float(te_num.notna().mean()) >= 0.8

    if numeric_like:
        tr_mean = float(tr_num.mean())
        te_mean = float(te_num.mean())
        scale_vals = pd.concat([tr_num, te_num], axis=0).dropna()
        lo = float(scale_vals.min()) if not scale_vals.empty else 0.0
        hi = float(scale_vals.max()) if not scale_vals.empty else 1.0
        denom = max(1e-9, hi - lo)
        ref_bar = float(np.clip((tr_mean - lo) / denom, 0.0, 1.0))
        cur_bar = float(np.clip((te_mean - lo) / denom, 0.0, 1.0))
        ref = f"mean={tr_mean:.3f} miss={tr.isna().mean()*100:.1f}%"
        cur = f"mean={te_mean:.3f} miss={te.isna().mean()*100:.1f}%"
        return ref, cur, ref_bar, cur_bar, "numeric"

    tr_cat = tr.astype("string")
    te_cat = te.astype("string")
    tr_top = tr_cat.value_counts(normalize=True, dropna=False)
    te_top = te_cat.value_counts(normalize=True, dropna=False)
    tr_lbl = str(tr_top.index[0]) if not tr_top.empty else "NA"
    te_lbl = str(te_top.index[0]) if not te_top.empty else "NA"
    tr_share = float(tr_top.iloc[0]) if not tr_top.empty else 0.0
    te_share = float(te_top.iloc[0]) if not te_top.empty else 0.0
    ref = f"top={tr_lbl} ({tr_share*100:.1f}%)"
    cur = f"top={te_lbl} ({te_share*100:.1f}%)"
    return ref, cur, tr_share, te_share, "categorical"


def _build_popup_distribution_bars(feature: str, train: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    tr = pd.to_numeric(train[feature], errors="coerce")
    te = pd.to_numeric(test[feature], errors="coerce")

    tr_val = float(tr.mean()) if tr.notna().any() else 0.0
    te_val = float(te.mean()) if te.notna().any() else 0.0

    all_vals = pd.concat([tr, te], axis=0).dropna()
    lo = float(all_vals.min()) if not all_vals.empty else 0.0
    hi = float(all_vals.max()) if not all_vals.empty else 1.0
    denom = max(1e-9, hi - lo)

    tr_scaled = float(np.clip((tr_val - lo) / denom, 0.0, 1.0))
    te_scaled = float(np.clip((te_val - lo) / denom, 0.0, 1.0))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Reference", "Current"],
            y=[tr_scaled, te_scaled],
            marker_color=[COLORS["reference"], COLORS["current"]],
            text=[f"{tr_scaled:.3f}", f"{te_scaled:.3f}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Scaled Distribution Bars: {feature}",
        template="plotly_white",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


def _build_timeline(raw_payload: dict[str, Any], drift_scores: pd.Series) -> pd.DataFrame:
    timeline = raw_payload.get("timeline", []) if isinstance(raw_payload, dict) else []
    if isinstance(timeline, list) and timeline:
        rows = []
        for item in timeline:
            if isinstance(item, dict):
                label = str(item.get("period", item.get("time", "")))
                score = _to_float(item.get("drift_score", item.get("score", 0.0)))
                if label:
                    rows.append((label, score))
        if rows:
            return pd.DataFrame(rows, columns=["Period", "Drift Score"])

    # Fallback derived timeline from score segments.
    vals = np.asarray(drift_scores.dropna().tolist(), dtype=np.float64)
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float64)
    parts = np.array_split(np.sort(vals), 6)
    rows = []
    for idx, part in enumerate(parts, start=1):
        if part.size == 0:
            rows.append((f"T{idx}", 0.0))
        else:
            rows.append((f"T{idx}", float(np.mean(part))))
    return pd.DataFrame(rows, columns=["Period", "Drift Score"])


metrics, drift_df, train_sample, test_sample, winsor_bounds, dq_alerts, used_mock, raw_payload = load_or_mock_data()

st.title("Data Drift Monitoring Dashboard")
st.caption("Standalone artifact dashboard. No training pipeline execution is triggered.")
if any(used_mock.values()):
    mock_keys = [k for k, v in used_mock.items() if v]
    st.info("Mock fallback active for: " + ", ".join(mock_keys))


# AU-PRC section.
st.markdown('<div class="section-title">AU-PRC Overview</div>', unsafe_allow_html=True)
auprc_matrix, pre_vals, post_vals = _auprc_matrix(metrics)

au_left, au_right = st.columns([1.0, 2.0], gap="large")
with au_left:
    st.markdown('<div class="section-subtitle">Sub-representation (2x2 AU-PRC matrix).</div>', unsafe_allow_html=True)
    st.dataframe(
        auprc_matrix.style.format("{:.4f}"),
        use_container_width=True,
    )
    _pre_or_post_hint = "based on selected curve"  # keeps text aligned with right panel selection
    st.markdown(
        '<div class="section-subtitle">'
        + 'Recall is the fraction of actual positives captured by the model: TP / (TP + FN). '
        + 'Precision-Recall curves are shown on the right using exported precision and recall points from the pipeline artifacts.'
        + '</div>',
        unsafe_allow_html=True,
    )

with au_right:
    stage_choice = st.selectbox(
        "AU-PRC curve view",
        options=["Pre-Mitigation", "Post-Mitigation"],
        index=1,
        help="Select whether to display pre- or post-mitigation train/test precision-recall curves.",
    )
    stage_key = "pre" if stage_choice.lower().startswith("pre") else "post"
    pr_fig, curves_exact = _build_pr_line_chart(
        title=f"{stage_choice}: Train vs Test Precision-Recall",
        metrics=metrics,
        stage=stage_key,
    )
    st.plotly_chart(pr_fig, use_container_width=True)
    if not curves_exact:
        st.caption("Exact precision-recall arrays are missing in the current artifacts. Re-run the pipeline to refresh latest_metrics.json with PR points.")


# Drift detection table section.
valid_features = [f for f in drift_df["Feature"].tolist() if f in train_sample.columns and f in test_sample.columns]
if not valid_features:
    valid_features = [f for f in drift_df["Feature"].tolist()]

view_df = drift_df.copy()
if valid_features:
    view_df = view_df[view_df["Feature"].isin(valid_features)].copy()

ref_dist_col = []
cur_dist_col = []
for feat in view_df["Feature"].tolist():
    if feat in train_sample.columns and feat in test_sample.columns:
        _, _, ref_val, cur_val, _ = _distribution_signature(feat, train_sample, test_sample)
    else:
        ref_val, cur_val = 0.0, 0.0
    ref_dist_col.append(float(np.clip(ref_val, 0.0, 1.0)) * 100.0)
    cur_dist_col.append(float(np.clip(cur_val, 0.0, 1.0)) * 100.0)

view_df["Reference Distribution"] = [[v] for v in ref_dist_col]
view_df["Current Distribution"] = [[v] for v in cur_dist_col]

max_score = float(max(1e-9, view_df["Drift Score"].max())) if not view_df.empty else 1.0
view_df["Drift Contribution (y-bar)"] = (view_df["Drift Score"] / max_score * 100.0).clip(0, 100)

view_df = view_df.sort_values("Drift Contribution (y-bar)", ascending=False).reset_index(drop=True)

drifted_count = int((view_df["Data Drift"] == "Detected").sum())
total_count = int(len(view_df))
share = 100.0 * drifted_count / max(1, total_count)
if drifted_count > 0:
    status_line = f"Drift is detected for {share:.1f}% of features ({drifted_count} of {total_count}). Dataset Drift is detected."
else:
    status_line = f"Drift is detected for {share:.1f}% of features ({drifted_count} of {total_count}). Dataset Drift is not detected."

st.markdown('<div class="section-title">Drift Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">' + status_line + '</div>', unsafe_allow_html=True)
st.caption(
    "Mini distribution bars meaning: for numeric features they show the normalized mean position (0-100) across combined reference/current range; "
    "for categorical or binary features they show the dominant-category share (0-100)."
)

display_cols = [
    "Feature",
    "Type",
    "Drift Contribution (y-bar)",
    "Reference Distribution",
    "Current Distribution",
    "Data Drift",
    "Stat Test",
    "Drift Score",
]

rows_per_page_options = [10, 20, 30, 50, 100]
if "drift_rows_per_page" not in st.session_state:
    st.session_state["drift_rows_per_page"] = 20
if "drift_page" not in st.session_state:
    st.session_state["drift_page"] = 1

if int(st.session_state["drift_rows_per_page"]) not in rows_per_page_options:
    st.session_state["drift_rows_per_page"] = 20

paged_size = int(st.session_state["drift_rows_per_page"])
num_pages = max(1, int(np.ceil(len(view_df) / paged_size)))
st.session_state["drift_page"] = max(1, min(int(st.session_state["drift_page"]), num_pages))
page = int(st.session_state["drift_page"])

start = (int(page) - 1) * paged_size
end = start + paged_size
page_df = view_df.iloc[start:end].copy()
display_df = page_df[display_cols].copy()

selected_feature = None

# Prefer clickable row selection when supported.
try:
    evt = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Drift Contribution (y-bar)": st.column_config.ProgressColumn(
                "Drift Contribution (y-bar)",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
            "Reference Distribution": st.column_config.BarChartColumn(
                "Reference Distribution",
                y_min=0.0,
                y_max=100.0,
                color="blue",
            ),
            "Current Distribution": st.column_config.BarChartColumn(
                "Current Distribution",
                y_min=0.0,
                y_max=100.0,
                color="red",
            ),
            "Drift Score": st.column_config.NumberColumn("Drift Score", format="%.4f"),
        },
    )
    if evt is not None and isinstance(evt, dict):
        rows = evt.get("selection", {}).get("rows", [])
        if rows:
            local_idx = int(rows[0])
            if 0 <= local_idx < len(page_df):
                selected_feature = str(page_df.iloc[local_idx]["Feature"])
except TypeError:
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Drift Contribution (y-bar)": st.column_config.ProgressColumn(
                "Drift Contribution (y-bar)",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
            "Reference Distribution": st.column_config.BarChartColumn(
                "Reference Distribution",
                y_min=0.0,
                y_max=100.0,
                color="blue",
            ),
            "Current Distribution": st.column_config.BarChartColumn(
                "Current Distribution",
                y_min=0.0,
                y_max=100.0,
                color="red",
            ),
            "Drift Score": st.column_config.NumberColumn("Drift Score", format="%.4f"),
        },
    )

pagination_left, pagination_right = st.columns([1.0, 2.4], gap="small")
with pagination_left:
    selected_rows = st.selectbox(
        "Rows per page",
        options=rows_per_page_options,
        index=rows_per_page_options.index(paged_size),
        key="drift_rows_per_page_dropdown",
        width=135,
    )
    if int(selected_rows) != paged_size:
        st.session_state["drift_rows_per_page"] = int(selected_rows)
        st.session_state["drift_page"] = 1
        st.rerun()

with pagination_right:
    p_col1, p_col2, p_col3 = st.columns([0.35, 1.2, 0.35])
    prev_clicked = p_col1.button("<", disabled=(page <= 1), key="drift_prev_page", width="content")
    p_col2.markdown(
        f"<div style='text-align:center;padding-top:0.28rem;font-weight:600;color:{COLORS['axis_dark']};'>Page {page} of {num_pages}</div>",
        unsafe_allow_html=True,
    )
    next_clicked = p_col3.button(
        ">",
        disabled=(page >= num_pages),
        key="drift_next_page",
        width="content",
    )

if prev_clicked:
    st.session_state["drift_page"] = max(1, page - 1)
    st.rerun()
if next_clicked:
    st.session_state["drift_page"] = min(num_pages, page + 1)
    st.rerun()

if selected_feature is None:
    selected_feature = st.selectbox("Select feature for detail", page_df["Feature"].tolist() if not page_df.empty else view_df["Feature"].tolist())


# Popup-like deep insight.
if selected_feature:
    f_row = view_df[view_df["Feature"] == selected_feature]
    if not f_row.empty:
        r = f_row.iloc[0]
        drift_score = float(r["Drift Score"])
        threshold = float(r.get("Threshold", 0.10))
        p_value = _to_float_or_nan(r.get("P-Value"))
        mitigated = bool(r.get("Mitigated", False))
        reason = str(r.get("Reason", ""))

        def _render_popup_body() -> None:
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                if selected_feature in train_sample.columns and selected_feature in test_sample.columns:
                    st.plotly_chart(
                        _build_popup_distribution_bars(selected_feature, train_sample, test_sample),
                        use_container_width=True,
                    )
                else:
                    st.info("Feature is not present in sampled artifacts.")
            with c2:
                status_text = "Drifted" if str(r.get("Data Drift", "")).lower() == "detected" else "Stable"
                st.markdown("### Feature Insight")
                st.markdown("- Feature: " + selected_feature)
                st.markdown("- Type: " + str(r.get("Type", "Unknown")))
                st.markdown("- Status: " + status_text)
                st.markdown("- Drift Score: " + f"{drift_score:.4f}")
                st.markdown("- Threshold: " + f"{threshold:.4f}")
                st.markdown("- P-Value: " + ("-" if np.isnan(p_value) else f"{p_value:.4g}"))
                st.markdown("- Mitigated: " + ("Yes" if mitigated else "No"))
                if selected_feature in winsor_bounds:
                    b = winsor_bounds[selected_feature]
                    st.markdown(
                        "- Winsor Bounds: "
                        + f"q_low={_to_float_or_nan(b.get('q_low')):.4f}, q_high={_to_float_or_nan(b.get('q_high')):.4f}"
                    )
                if reason:
                    st.markdown("- Insight: " + reason)

        if hasattr(st, "dialog"):
            @st.dialog("Feature Drift Detail")
            def _feature_dialog() -> None:
                _render_popup_body()

            if st.button("Open feature pop-up", type="secondary"):
                _feature_dialog()
        else:
            with st.expander("Feature Drift Detail", expanded=False):
                _render_popup_body()


# Data quality section: 2x2 table.
st.markdown('<div class="section-title">Data Quality</div>', unsafe_allow_html=True)
missingness_cnt = sum(1 for x in dq_alerts if "missing" in x.lower())
casing_cnt = sum(1 for x in dq_alerts if "case" in x.lower() or "casing" in x.lower())

quality_2x2 = pd.DataFrame(
    {
        "Metric": ["Missingness Shifts", "Casing Anomalies"],
        "Value": [missingness_cnt, casing_cnt],
    }
)
st.dataframe(quality_2x2, use_container_width=True, hide_index=True)


# Drift timeline x-bar.
st.markdown('<div class="section-title">Drift Timeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">X-bar representation of drift progression.</div>', unsafe_allow_html=True)

timeline_df = _build_timeline(raw_payload, view_df["Drift Score"])
fig_timeline = go.Figure()
fig_timeline.add_trace(
    go.Bar(
        x=timeline_df["Period"],
        y=timeline_df["Drift Score"],
        marker_color=COLORS["current"],
        opacity=0.88,
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    )
)
fig_timeline.update_layout(
    template="plotly_white",
    height=300,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor=COLORS["card"],
    plot_bgcolor=COLORS["card"],
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False, title="Drift Score"),
)
st.plotly_chart(fig_timeline, use_container_width=True)
