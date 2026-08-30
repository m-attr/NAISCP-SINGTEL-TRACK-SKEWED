from __future__ import annotations


PALETTE = {
    "primary": "#2563EB",
    "current": "#2563EB",
    "historical": "#94A3B8",
    "historical_light": "#B2BCC7",
    "keep": "#526579",
    "repair": "#16A36E",
    "drop": "#DC3F45",
    "text": "#202731",
    "muted": "#64748B",
    "border": "#D8E0E8",
    "border_strong": "#CBD5E1",
    "table_header": "#F1F5F9",
    "selected": "#EAF2FF",
    "background": "#FFFFFF",
    "surface_subtle": "#F8FAFC",
}

SPACING = {
    "xs": "8px",
    "sm": "12px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
}

ACTION_ORDER = ("Keep", "Repair", "Drop")
ACTION_COLORS = {
    "Keep": PALETTE["keep"],
    "Repair": PALETTE["repair"],
    "Drop": PALETTE["drop"],
}

SEVERITY_HELP = (
    "Severity compares the current drift with the feature's historical movement "
    "using a stabilised ratio. Higher values indicate stronger evidence of "
    "unusually large drift."
)

PLOT_BACKGROUND = "rgba(0,0,0,0)"
PLOT_GRID = "#E8EDF2"
