"""SURDview — Streamlit dashboard for SURD causal decomposition.

Method: Martínez-Sánchez, Á., Arranz, G., Lozano-Durán, A. (2024)
        'Decomposing causality into its synergistic, unique, and
        redundant components', Nature Communications, 15, 9296.
"""

import streamlit as st
import pandas as pd

from src.components.data_ingestion import load_csv, dataset_summary
from src.components.explanation import generate_explanation
from src.components.evaluation import run_evaluation
from src.pipeline.analysis_pipeline import run_analysis
from src.pipeline.export_pipeline import result_to_json, result_to_csv, export_chart_html
from src.plots import (
    make_overview_fig, make_breakdown_fig, make_pie_fig,
    make_leak_fig, make_evaluation_fig,
)

# ── Page config ──
st.set_page_config(page_title="SURDview", page_icon="🔍", layout="wide")

st.title("🔍 SURDview")
st.caption(
    "Causal decomposition into synergistic, unique, and redundant components "
    "— powered by SURD (Martínez-Sánchez et al., 2024)."
)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("1 · Upload data")
    uploaded = st.file_uploader("Choose a CSV file (time-ordered rows)", type=["csv"])

    use_example = st.checkbox("Use synergistic collider example",
                              value=not bool(uploaded))

    df: pd.DataFrame | None = None

    if uploaded:
        df = load_csv(uploaded)
    elif use_example:
        df = load_csv("data/synergistic_collider.csv")

    if df is not None:
        summary = dataset_summary(df)
        st.markdown(
            f"**Rows:** {summary['rows']}  \n"
            f"**Columns:** {summary['columns']}  \n"
            f"**Missing cells:** {summary['missing_cells']}"
        )

        numeric_only = st.toggle("Numeric columns only", value=True)
        if numeric_only:
            available_cols = summary["numeric_columns"]
        else:
            available_cols = list(df.columns)

        st.divider()
        st.header("2 · Choose columns")

        if len(available_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            st.stop()

        target = st.selectbox("Target variable (whose future to predict)", options=available_cols)
        agent_options = available_cols
        agents = st.multiselect(
            "Agent variables (whose past may cause the target)",
            options=agent_options,
            default=agent_options[:min(3, len(agent_options))],
            max_selections=6,
            help="Include the target itself to measure self-causation.",
        )

        st.divider()
        st.header("3 · Settings")
        tau = st.slider("Time lag (τ)", min_value=1, max_value=20, value=1,
                        help="How many time steps into the future to predict.")
        nbins = st.slider("Histogram bins", min_value=2, max_value=20, value=8,
                          help="Number of bins per variable for probability estimation.")
        missing_strategy = st.selectbox(
            "Missing-values strategy",
            options=["drop", "mean"],
            format_func=lambda x: "Drop rows" if x == "drop" else "Fill with mean",
        )

        st.divider()
        run_clicked = st.button("🚀 Run SURD Analysis", use_container_width=True,
                                type="primary")
    else:
        run_clicked = False

# ══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════
if df is None:
    st.info("Upload a CSV or tick **Use synergistic collider example** to start.")
    st.stop()

if len(agents) < 1:
    st.warning("Select at least **1 agent variable** before running.")
    st.stop()

# Run analysis and store in session state.
if run_clicked:
    with st.spinner("Running SURD causal decomposition…"):
        try:
            result = run_analysis(
                df, target=target, agents=agents,
                params={
                    "tau": tau,
                    "nbins": nbins,
                    "missing_strategy": missing_strategy,
                    "explanation_mode": "plain",
                },
            )
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

result = st.session_state.get("result")

if result is None:
    st.info("Configure settings in the sidebar and click **Run SURD Analysis**.")
    st.stop()

# ── Tabs ──
tab_overview, tab_details, tab_explain, tab_export, tab_eval = st.tabs(
    ["📊 Overview", "🔬 Breakdown", "💡 Explanation", "📥 Export", "🧪 Evaluation"]
)

# ── Overview tab ──
with tab_overview:
    st.plotly_chart(make_overview_fig(result), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total unique", f"{result['total_unique']:.4f} bits")
    c2.metric("Total redundant", f"{result['total_redundant']:.4f} bits")
    c3.metric("Total synergy", f"{result['total_synergy']:.4f} bits")
    c4.metric("Info leak", f"{result['info_leak']:.1%}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(make_pie_fig(result), use_container_width=True)
    with col_right:
        st.plotly_chart(make_leak_fig(result), use_container_width=True)

    st.caption(f"**Method:** {result.get('method', 'N/A')}")

# ── Breakdown tab ──
with tab_details:
    st.plotly_chart(make_breakdown_fig(result), use_container_width=True)

    st.subheader("Component details")
    col_u, col_r, col_s = st.columns(3)

    with col_u:
        st.markdown("**Unique causalities**")
        for name, val in result.get("unique", {}).items():
            st.markdown(f"- U({name}): {val:.6f}")

    with col_r:
        st.markdown("**Redundant causalities**")
        for name, val in result.get("redundant_breakdown", {}).items():
            st.markdown(f"- R({name.replace('|',',')}): {val:.6f}")

    with col_s:
        st.markdown("**Synergistic causalities**")
        for name, val in result.get("synergy_breakdown", {}).items():
            st.markdown(f"- S({name.replace('|',',')}): {val:.6f}")

# ── Explanation tab ──
with tab_explain:
    mode = st.radio("Explanation style:", ["Plain", "Technical"], horizontal=True)
    explanation = generate_explanation(result, mode=mode.lower())
    st.markdown(explanation)

# ── Export tab ──
with tab_export:
    col_a, col_b = st.columns(2)
    with col_a:
        json_str = result_to_json(result)
        st.download_button("⬇️ Download JSON", data=json_str,
                           file_name="surd_results.json", mime="application/json")
    with col_b:
        csv_str = result_to_csv(result)
        st.download_button("⬇️ Download CSV summary", data=csv_str,
                           file_name="surd_results.csv", mime="text/csv")

    st.divider()
    if st.button("💾 Save overview chart to artifacts/"):
        fig = make_overview_fig(result)
        path = export_chart_html(fig, "overview_chart.html")
        st.success(f"Saved to `{path}`")

    with st.expander("Preview JSON"):
        st.code(json_str, language="json")

# ── Evaluation tab ──
with tab_eval:
    st.subheader("Sanity-check on benchmark datasets")
    st.markdown(
        "These four datasets have **known causal structures** from the "
        "[SURD paper](https://doi.org/10.1038/s41467-024-53373-4). "
        "Running SURD on them verifies the method is working correctly."
    )

    if st.button("🧪 Run evaluation", type="primary"):
        with st.spinner("Running SURD on mediator, confounder, synergistic and redundant colliders…"):
            try:
                eval_results = run_evaluation(tau=tau, nbins=nbins)
                st.session_state["eval_results"] = eval_results
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    eval_results = st.session_state.get("eval_results")
    if eval_results:
        st.plotly_chart(make_evaluation_fig(eval_results), use_container_width=True)

        for er in eval_results:
            icon = "✅" if er["passed"] else "⚠️"
            with st.expander(f"{icon} {er['dataset']} — {er['expected']}"):
                r = er["result"]
                st.markdown(
                    f"- **Total unique:** {r['total_unique']:.4f}\n"
                    f"- **Total redundant:** {r['total_redundant']:.4f}\n"
                    f"- **Total synergy:** {r['total_synergy']:.4f}\n"
                    f"- **Info leak:** {r['info_leak']:.1%}\n"
                    f"- **Pass:** {'Yes' if er['passed'] else 'No'}"
                )

        passed = sum(1 for e in eval_results if e["passed"])
        st.markdown(f"**Score: {passed}/{len(eval_results)} tests passed.**")
    else:
        st.info("Click **Run evaluation** to test the method on benchmark data.")
