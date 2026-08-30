from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from dashboard.theme import PALETTE, SPACING


def inject_styles() -> None:
    colour_tokens = "\n".join(
        f"          --dd-{name.replace('_', '-')}: {value};"
        for name, value in PALETTE.items()
    )
    spacing_tokens = "\n".join(
        f"          --space-{name}: {value};" for name, value in SPACING.items()
    )
    st.markdown(
        f"""
        <style>
        :root {{
{colour_tokens}
{spacing_tokens}
        }}
        .stApp {{ background: var(--dd-background); color: var(--dd-text); }}
        [data-testid="stHeader"] {{ background: rgba(255,255,255,.96); }}
        [data-testid="stSidebar"] {{
          background: var(--dd-surface-subtle);
          border-right: 1px solid var(--dd-border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--dd-text); }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {{
          border-left: 3px solid transparent;
          border-radius: var(--space-xs);
          padding: var(--space-sm);
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{
          background: var(--dd-selected);
          border-left-color: var(--dd-primary);
        }}
        .block-container {{
          max-width: 1240px;
          padding-top: var(--space-xl);
          padding-bottom: 48px;
        }}
        .dd-sidebar-title {{
          color: var(--dd-text); font-size: 1.1rem; font-weight: 720;
          letter-spacing: -.02em; padding: var(--space-xs) var(--space-sm) var(--space-md);
        }}
        .dd-footer {{
          color: var(--dd-muted); border-top: 1px solid var(--dd-border);
          margin-top: var(--space-xl); padding: var(--space-md) 0 var(--space-xs);
          text-align: center; font-size: .75rem;
        }}
        .dd-page-header {{ margin-bottom: var(--space-lg); }}
        .dd-title {{ color: var(--dd-text); font-size: 2rem; font-weight: 700;
          letter-spacing: -.03em; line-height: 1.12; margin: 0; }}
        .dd-subtitle {{ color: var(--dd-muted); max-width: 760px; font-size: .96rem;
          line-height: 1.5; margin: var(--space-xs) 0 0; }}
        .dd-card {{
          --dd-card-accent: var(--dd-primary); background: var(--dd-background);
          border: 1px solid var(--dd-border); border-top: 4px solid var(--dd-card-accent);
          border-radius: var(--space-xs); padding: var(--space-sm) var(--space-md);
          min-height: 96px;
        }}
        .dd-card-blue {{ --dd-card-accent: var(--dd-primary); }}
        .dd-card-keep {{ --dd-card-accent: var(--dd-keep); }}
        .dd-card-repair {{ --dd-card-accent: var(--dd-repair); }}
        .dd-card-drop {{ --dd-card-accent: var(--dd-drop); }}
        .dd-card-historical {{ --dd-card-accent: var(--dd-historical); }}
        .dd-card-label {{ color: var(--dd-muted); font-size: .76rem; font-weight: 600; }}
        .dd-card-value {{ color: var(--dd-text); font-size: 1.55rem; line-height: 1.2;
          font-weight: 700; margin-top: var(--space-xs); }}
        .dd-card-note {{ color: var(--dd-muted); font-size: .7rem;
          margin-top: var(--space-xs); min-height: 1em; }}
        .dd-section-wrap {{ margin: var(--space-xl) 0 var(--space-sm); }}
        .dd-section {{ color: var(--dd-text); font-size: 1.16rem; font-weight: 680;
          letter-spacing: -.015em; margin: 0; }}
        .dd-section-note {{ color: var(--dd-muted); font-size: .88rem;
          margin: var(--space-xs) 0 0; }}
        .dd-chart-heading, .dd-detail-section-heading {{
          color: var(--dd-text); font-size: 1rem; font-weight: 680;
          letter-spacing: -.01em;
        }}
        .dd-chart-heading {{ margin: 0 0 var(--space-sm); }}
        .dd-detail-section-heading {{ margin: var(--space-lg) 0 var(--space-sm); }}
        .dd-chart-caption {{ color: var(--dd-muted); font-size: .78rem;
          line-height: 1.5; margin: var(--space-xs) 0 0; }}
        .dd-action {{ display: inline-block; border-radius: 5px;
          padding: 4px var(--space-xs); font-size: .68rem; font-weight: 700;
          letter-spacing: .025em; }}
        .dd-action-keep {{ color: var(--dd-keep); background: var(--dd-table-header); }}
        .dd-action-repair {{ color: var(--dd-repair); background: #E9F8F1; }}
        .dd-action-drop {{ color: var(--dd-drop); background: #FDECEE; }}
        .dd-feature-heading {{ display: flex; align-items: center; justify-content: space-between;
          gap: var(--space-md); border: 1px solid var(--dd-border);
          border-left: 4px solid var(--dd-primary); border-radius: var(--space-xs);
          margin-top: 0; padding: var(--space-sm) var(--space-md);
          background: var(--dd-selected); }}
        .dd-feature-title {{ color: var(--dd-text); font-size: 1.35rem; font-weight: 680;
          letter-spacing: -.02em; margin: 0; }}
        .dd-legend {{ color: var(--dd-muted); font-size: .78rem; line-height: 1.55;
          margin: var(--space-xs) 0 var(--space-lg); }}
        .st-key-explorer-filters {{ margin-bottom: var(--space-sm); }}
        .st-key-explorer-detail-metrics {{
          margin-top: var(--space-sm); margin-bottom: var(--space-lg);
        }}
        .dd-direction {{ border: 1px solid var(--dd-border); border-radius: var(--space-xs);
          padding: var(--space-sm) var(--space-md); background: var(--dd-surface-subtle); }}
        .dd-direction-row {{ display: grid; grid-template-columns: minmax(150px,1fr) auto auto;
          gap: var(--space-sm); padding: 4px 0; color: var(--dd-muted); font-size: .86rem; }}
        .dd-direction-value {{ color: var(--dd-text); font-variant-numeric: tabular-nums;
          font-weight: 650; }}
        .dd-direction-arrow {{ color: var(--dd-primary); font-weight: 700;
          min-width: 2rem; text-align: center; }}
        .dd-direction-note {{ color: var(--dd-muted); font-size: .78rem;
          margin-top: var(--space-xs); }}
        .dd-detail-list {{ border-top: 1px solid var(--dd-border); }}
        .dd-detail-row {{ display: grid; grid-template-columns: minmax(120px, .28fr) 1fr;
          gap: var(--space-md); border-bottom: 1px solid var(--dd-border);
          padding: var(--space-xs) 2px; font-size: .86rem; }}
        .dd-detail-key {{ color: var(--dd-muted); }}
        .dd-detail-value {{ color: var(--dd-text); }}
        .dd-placeholder {{ min-height: 315px; border: 1px dashed var(--dd-border-strong);
          border-radius: var(--space-xs); display: flex; align-items: center;
          justify-content: center; text-align: center; color: var(--dd-muted);
          background: var(--dd-surface-subtle); font-size: .86rem; padding: var(--space-md); }}
        [data-testid="stMetric"] {{ background: var(--dd-background);
          border: 1px solid var(--dd-border); border-radius: var(--space-xs);
          padding: var(--space-sm) var(--space-md); }}
        [data-testid="stDataFrame"] {{ border: 1px solid var(--dd-border-strong);
          border-radius: var(--space-xs); overflow: hidden; }}
        [data-testid="stPlotlyChart"] {{ border: 1px solid var(--dd-border);
          border-radius: var(--space-xs); }}
        div[data-baseweb="select"] > div, [data-testid="stTextInput"] input {{
          border-radius: var(--space-xs);
        }}
        @media (max-width: 900px) {{
          .block-container {{ padding-top: var(--space-lg); }}
          .dd-direction-row {{ grid-template-columns: 1fr auto auto; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        '<div class="dd-page-header">'
        f'<h1 class="dd-title">{escape(title)}</h1>'
        f'<p class="dd-subtitle">{escape(subtitle)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    note: str = "",
    accent: str = "blue",
) -> None:
    safe_accent = (
        accent
        if accent in {"blue", "keep", "repair", "drop", "historical"}
        else "blue"
    )
    st.markdown(
        f'<div class="dd-card dd-card-{safe_accent}">'
        f'<div class="dd-card-label">{escape(label)}</div>'
        f'<div class="dd-card-value">{escape(value)}</div>'
        f'<div class="dd-card-note">{escape(note)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        '<div class="dd-section-wrap">'
        f'<div class="dd-section">{escape(title)}</div>'
        + (f'<p class="dd-section-note">{escape(note)}</p>' if note else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def detail_heading(title: str, *, separated: bool = False) -> None:
    css_class = "dd-detail-section-heading" if separated else "dd-chart-heading"
    st.markdown(
        f'<div class="{css_class}">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def chart_caption(text: str) -> None:
    st.markdown(
        f'<p class="dd-chart-caption">{escape(text)}</p>',
        unsafe_allow_html=True,
    )


def action_class(display_action: str) -> str:
    return {
        "REPAIR": "repair",
        "DROP": "drop",
        "KEEP": "keep",
    }.get(display_action, "keep")


def action_badge(display_action: str) -> str:
    css = action_class(display_action)
    return (
        f'<span class="dd-action dd-action-{css}">'
        f'{escape(display_action.title())}</span>'
    )


def direction_indicator(feature: dict[str, Any]) -> None:
    evidence = feature["evidence"]
    train_value = float(evidence["orientation_train"])
    test_value = float(evidence["orientation_test"])
    train_arrow = "←" if train_value < 0 else "→"
    test_arrow = "←" if test_value < 0 else "→"
    anchors = int(evidence["orientation_anchor_count"])
    st.markdown(
        '<div class="dd-direction">'
        '<div class="dd-direction-row"><span>Recent training direction</span>'
        f'<span class="dd-direction-value">{train_value:+.3f}</span>'
        f'<span class="dd-direction-arrow">{train_arrow}</span></div>'
        '<div class="dd-direction-row"><span>Current direction</span>'
        f'<span class="dd-direction-value">{test_value:+.3f}</span>'
        f'<span class="dd-direction-arrow">{test_arrow}</span></div>'
        f'<div class="dd-direction-note">Direction reversed · {anchors} stable references</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def detail_rows(rows: list[tuple[str, Any]]) -> None:
    body = "".join(
        '<div class="dd-detail-row">'
        f'<div class="dd-detail-key">{escape(str(key))}</div>'
        f'<div class="dd-detail-value">{escape(str(value))}</div>'
        '</div>'
        for key, value in rows
    )
    st.markdown(f'<div class="dd-detail-list">{body}</div>', unsafe_allow_html=True)
