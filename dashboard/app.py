from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.components.ui import inject_styles
from dashboard.data.artifacts import (
    ArtifactError,
    load_dashboard_run,
    resolve_artifact_path,
)
from dashboard.pages import drift_explorer, overview, performance_runtime

st.set_page_config(
    page_title="DataDrift",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_styles()


@st.cache_data(show_spinner=False)
def _cached_load(path: str, modified_ns: int) -> dict:
    del modified_ns
    return load_dashboard_run(path)


artifact_path = resolve_artifact_path()
try:
    modified_ns = artifact_path.stat().st_mtime_ns if artifact_path.is_file() else -1
    dashboard_run = _cached_load(str(artifact_path), modified_ns)
except ArtifactError as exc:
    st.title("Data unavailable for this run")
    st.caption(str(exc))
    st.write(
        "Build a real, bounded dashboard artifact first. The application will not fill "
        "missing panels with generated data."
    )
    st.code(
        "python tools/build_dashboard_artifacts.py "
        "--train_data_filepath train.csv --test_data_filepath test.csv "
        "--run_artifact_dir <completed-run-directory> "
        "--output_filepath dashboard_artifacts/dashboard_run.json",
        language="bash",
    )
    st.stop()

pages = [
    st.Page(
        lambda: overview.render(dashboard_run),
        title="Overview",
        url_path="overview",
        default=True,
    ),
    st.Page(
        lambda: drift_explorer.render(dashboard_run),
        title="Drift Explorer",
        url_path="drift-explorer",
    ),
    st.Page(
        lambda: performance_runtime.render(dashboard_run),
        title="Performance & Runtime",
        url_path="performance-runtime",
    ),
]
navigation = st.navigation(pages, position="hidden")
with st.sidebar:
    st.markdown(
        '<div class="dd-sidebar-title">DataDrift - Skewed</div>',
        unsafe_allow_html=True,
    )
    st.page_link(pages[0], label="Overview", width="stretch")
    st.page_link(pages[1], label="Drift Explorer", width="stretch")
    st.page_link(pages[2], label="Performance & Runtime", width="stretch")
navigation.run()
st.markdown(
    '<footer class="dd-footer">© 2026 Team Skewed · Matt &amp; Angelyn · NAISC | Singtel</footer>',
    unsafe_allow_html=True,
)
