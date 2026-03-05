
import streamlit as st
import pandas as pd

from src.components.data_ingestion import load_csv, dataset_summary
from src.pipeline.analysis_pipeline import run_analysis
from src.utils import make_overview_fig, make_heatmap_fig, result_to_json

# -- Page config --
st.set_page_config(
    page_title="SURDview",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 SURDview")
st.caption("Explore how much each source variable uniquely, redundantly, or synergistically tells you about a target.")

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("1 · Upload data")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    # Let users try the built-in example.
    use_example = st.checkbox("Use example weather dataset", value=not bool(uploaded))

    df: pd.DataFrame | None = None

    if uploaded:
        df = load_csv(uploaded)
    elif use_example:
        df = load_csv("data/example_small.csv")

    if df is not None:
        summary = dataset_summary(df)
        st.markdown(
            f"**Rows:** {summary['rows']}  \n"
            f"**Columns:** {summary['columns']}  \n"
            f"**Missing cells:** {summary['missing_cells']}"
        )

        st.divider()
        st.header("2 · Choose columns")

        numeric_cols = summary["numeric_columns"]
        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns (1 target + 2 sources).")
            st.stop()

        target = st.selectbox("Target (Y)", options=numeric_cols)
        available_sources = [c for c in numeric_cols if c != target]
        sources = st.multiselect(
            "Sources (X)",
            options=available_sources,
            default=available_sources[:2],
        )

        st.divider()
        st.header("3 · Settings")
        bins = st.slider("Number of bins", min_value=2, max_value=20, value=5)
        missing_strategy = st.selectbox(
            "Missing-values strategy",
            options=["drop", "mean"],
            format_func=lambda x: "Drop rows" if x == "drop" else "Fill with mean",
        )

        st.divider()
        run_clicked = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    else:
        run_clicked = False

# ======================================================================
# MAIN AREA
# ======================================================================
if df is None:
    st.info("Upload a CSV or tick 'Use example weather dataset' in the sidebar to get started.")
    st.stop()

if len(sources) < 2:
    st.warning("Select at least **2 source columns** in the sidebar before running.")
    st.stop()

# Keep results in session state so they survive reruns.
if run_clicked:
    with st.spinner("Running SURD analysis…"):
        result = run_analysis(
            df,
            sources=sources,
            target=target,
            params={"bins": bins, "missing_strategy": missing_strategy},
        )
        st.session_state["result"] = result

result = st.session_state.get("result")

if result is None:
    st.info("Configure your settings in the sidebar and click **Run Analysis**.")
    st.stop()

# -- Tabs --
tab_overview, tab_details, tab_explain, tab_export = st.tabs(
    ["📊 Overview", "🗺️ Details", "💡 Explanation", "📥 Export"]
)

with tab_overview:
    st.plotly_chart(make_overview_fig(result), use_container_width=True)

    # Quick numbers at a glance.
    cols = st.columns(3)
    cols[0].metric("Total Unique", f"{sum(result['unique'].values()):.4f} bits")
    cols[1].metric("Redundant", f"{result['redundant']:.4f} bits")
    cols[2].metric("Synergy", f"{result['synergy']:.4f} bits")

with tab_details:
    if result["pairwise_synergy"]:
        st.plotly_chart(make_heatmap_fig(result), use_container_width=True)
    else:
        st.info("Pairwise synergy heatmap is available when you select more than 2 source columns.")

with tab_explain:
    st.markdown(result["explanation"])

with tab_export:
    json_str = result_to_json(result)
    st.download_button(
        label="⬇️ Download results as JSON",
        data=json_str,
        file_name="surd_results.json",
        mime="application/json",
    )
    with st.expander("Preview JSON"):
        st.code(json_str, language="json")
