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
from src.components.surd_engine import run_lag_sweep
from src.components.chatbot import chat, get_chatbot_mode
from src.components.significance import permutation_test, interpret_pvalue
from src.components.comparison import compare_surd_te, explain_comparison
from src.components.interpretation import (
    interpret_unique, interpret_redundant, interpret_synergy, interpret_leak,
    interpret_unique_per_agent, interpret_pair_synergy, interpret_pair_redundant,
    interpret_pvalue as interpret_p, interpret_te_vs_unique, interpret_lag_curve,
)
from src.pipeline.analysis_pipeline import run_analysis
from src.pipeline.export_pipeline import result_to_json, result_to_csv, export_chart_html
from src.plots import (
    make_overview_fig, make_breakdown_fig, make_pie_fig,
    make_leak_fig, make_evaluation_fig, make_lag_sweep_fig,
    make_heatmap_fig, make_significance_fig,
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
tab_overview, tab_details, tab_heatmap, tab_sweep, tab_sig, tab_compare, tab_explain, tab_chat, tab_export, tab_eval = st.tabs(
    ["📊 Overview", "🔬 Breakdown", "🗺️ Heatmaps", "📈 Lag sweep", "📐 Significance",
     "🆚 vs Transfer Entropy", "💡 Explanation", "💬 Chat", "📥 Export", "🧪 Evaluation"]
)

# ── Overview tab ──
with tab_overview:
    st.plotly_chart(make_overview_fig(result), use_container_width=True)

    total = result["total_unique"] + result["total_redundant"] + result["total_synergy"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total unique", f"{result['total_unique']:.4f} bits",
              help=interpret_unique(result['total_unique'], total))
    c2.metric("Total redundant", f"{result['total_redundant']:.4f} bits",
              help=interpret_redundant(result['total_redundant'], total))
    c3.metric("Total synergy", f"{result['total_synergy']:.4f} bits",
              help=interpret_synergy(result['total_synergy'], total))
    c4.metric("Info leak", f"{result['info_leak']:.1%}",
              help=interpret_leak(result['info_leak']))

    with st.expander("📖 What do these numbers mean?"):
        st.markdown(f"""
        **Total unique** ({result['total_unique']:.4f} bits): {interpret_unique(result['total_unique'], total)}

        **Total redundant** ({result['total_redundant']:.4f} bits): {interpret_redundant(result['total_redundant'], total)}

        **Total synergy** ({result['total_synergy']:.4f} bits): {interpret_synergy(result['total_synergy'], total)}

        **Info leak** ({result['info_leak']:.1%}): {interpret_leak(result['info_leak'])}
        """)

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

    # Build three dataframes for the three component types.
    import pandas as pd

    unique_data = result.get("unique", {})
    redundant_data = result.get("redundant_breakdown", {})
    synergy_data = result.get("synergy_breakdown", {})

    if unique_data:
        st.markdown("**Unique causalities**")
        u_df = pd.DataFrame([
            {"Agent": name, "Bits": val}
            for name, val in unique_data.items()
        ])
        st.dataframe(
            u_df.style.format({"Bits": "{:.6f}"}),
            use_container_width=True, hide_index=True,
        )

    if redundant_data:
        st.markdown("**Redundant causalities**")
        r_df = pd.DataFrame([
            {"Combination": name.replace("|", ", "), "Bits": val}
            for name, val in redundant_data.items()
        ])
        st.dataframe(
            r_df.style.format({"Bits": "{:.6f}"}),
            use_container_width=True, hide_index=True,
        )

    if synergy_data:
        st.markdown("**Synergistic causalities**")
        s_df = pd.DataFrame([
            {"Combination": name.replace("|", ", "), "Bits": val}
            for name, val in synergy_data.items()
        ])
        st.dataframe(
            s_df.style.format({"Bits": "{:.6f}"}),
            use_container_width=True, hide_index=True,
        )

    with st.expander("📖 How to read these tables"):
        st.markdown(
            "**Unique causalities** show how much each agent predicts the target on its own. "
            "A high value means that agent carries information no other agent has. A near-zero "
            "value means the agent's influence is either shared with others or only emerges "
            "in combination."
        )
        st.markdown(
            "**Redundant causalities** show information that is duplicated across agents. "
            "If R(A, B) is high, knowing A tells you what B would tell you. You could often "
            "drop one without losing predictive power."
        )
        st.markdown(
            "**Synergistic causalities** show information that only exists when agents are "
            "observed together. If S(A, B) is high but U(A) and U(B) are both low, the agents "
            "are individually uninformative but their combination reveals something about the target."
        )

# ── Heatmap tab ──
with tab_heatmap:
    if result.get("redundant_breakdown") and result.get("synergy_breakdown"):
        hm_choice = st.radio(
            "Show:", ["Synergy heatmap", "Redundancy heatmap"],
            horizontal=True,
        )
        if hm_choice == "Synergy heatmap":
            st.plotly_chart(make_heatmap_fig(result, "synergy_breakdown"),
                            use_container_width=True)
        else:
            st.plotly_chart(make_heatmap_fig(result, "redundant_breakdown"),
                            use_container_width=True)

        # Highlight top pairs.
        syn = result.get("synergy_breakdown", {})
        red = result.get("redundant_breakdown", {})
        pair_syn = {k: v for k, v in syn.items() if "|" in k}
        pair_red = {k: v for k, v in red.items() if "|" in k}

        if pair_syn and pair_red:
            c1, c2 = st.columns(2)
            top_syn = max(pair_syn, key=pair_syn.get)
            top_red = max(pair_red, key=pair_red.get)
            c1.metric("Top synergistic pair",
                       top_syn.replace("|", " + "),
                       f"{pair_syn[top_syn]:.4f} bits")
            c2.metric("Most redundant pair",
                       top_red.replace("|", " + "),
                       f"{pair_red[top_red]:.4f} bits")

            with st.expander("📖 What do these pairs mean?"):
                st.markdown(
                    f"**Top synergistic pair ({top_syn.replace('|', ' + ')}):** "
                    f"{interpret_pair_synergy(top_syn, pair_syn[top_syn], (top_syn, pair_syn[top_syn]))}"
                )
                st.markdown(
                    f"**Most redundant pair ({top_red.replace('|', ' + ')}):** "
                    f"{interpret_pair_redundant(top_red, pair_red[top_red], (top_red, pair_red[top_red]))}"
                )
    else:
        st.info("Run analysis with 2 or more agents to see pairwise heatmaps.")

# ── Lag sweep tab ──
with tab_sweep:
    st.subheader("How does the causal decomposition change with lag?")
    st.markdown(
        "Run SURD at multiple tau values to see how the unique, "
        "redundant, and synergistic components change over time."
    )

    col_min, col_max, col_step = st.columns(3)
    with col_min:
        sweep_min = st.number_input("Min tau", min_value=1, value=1)
    with col_max:
        sweep_max = st.number_input("Max tau", min_value=2, value=10)
    with col_step:
        sweep_step = st.number_input("Step", min_value=1, value=1)

    if st.button("📈 Run lag sweep", type="primary"):
        tau_list = list(range(int(sweep_min), int(sweep_max) + 1, int(sweep_step)))
        with st.spinner(f"Running SURD at {len(tau_list)} tau values..."):
            try:
                sweep = run_lag_sweep(
                    df, target=target, agents=agents,
                    tau_list=tau_list, nbins=nbins)
                st.session_state["sweep"] = sweep
            except Exception as exc:
                st.error(f"Lag sweep failed: {exc}")

    sweep = st.session_state.get("sweep")
    if sweep:
        st.plotly_chart(make_lag_sweep_fig(sweep), use_container_width=True)

        with st.expander("📖 What does this curve tell you?"):
            st.markdown(interpret_lag_curve(sweep))

        st.markdown("**Summary table**")
        import pandas as pd
        sweep_df = pd.DataFrame(sweep)
        sweep_df.columns = ["Tau", "Unique", "Redundant", "Synergy", "Leak"]
        st.dataframe(sweep_df.style.format({
            "Unique": "{:.4f}", "Redundant": "{:.4f}",
            "Synergy": "{:.4f}", "Leak": "{:.4f}"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("Set the tau range above and click **Run lag sweep**.")

# ── Significance tab ──
with tab_sig:
    st.subheader("Statistical significance via permutation testing")
    st.markdown(
        "This shuffles each agent's time series many times to break causal "
        "links while preserving distributions. By running SURD on shuffled "
        "data, we build a null distribution for each component and compute "
        "p-values for the observed result."
    )

    n_perms = st.slider("Number of permutations", min_value=20, max_value=500,
                        value=100, step=20,
                        help="More permutations = more reliable p-values but slower.")

    if st.button("📐 Run permutation test", type="primary"):
        with st.spinner(f"Running {n_perms} permutations (this takes a minute)..."):
            try:
                sig = permutation_test(
                    df=df, target=target, agents=agents,
                    tau=tau, nbins=nbins, n_permutations=n_perms,
                )
                st.session_state["sig_result"] = sig
            except Exception as exc:
                st.error(f"Permutation test failed: {exc}")

    sig = st.session_state.get("sig_result")
    if sig:
        st.plotly_chart(make_significance_fig(sig), use_container_width=True)

        st.markdown("### Results")
        c1, c2, c3, c4 = st.columns(4)
        for col, name in zip([c1, c2, c3, c4],
                              ["unique", "redundant", "synergy", "leak"]):
            obs = sig["observed"][name]
            p = sig["p_values"][name]
            null_mean = sig["null_means"][name]
            label = name.capitalize()
            with col:
                st.metric(
                    label,
                    f"{obs:.4f}",
                    f"p = {p:.3f}",
                    delta_color="off",
                )
                st.caption(f"Null mean: {null_mean:.4f}")

        st.markdown("### Interpretation")
        for name in ["unique", "redundant", "synergy", "leak"]:
            obs = sig["observed"][name]
            p = sig["p_values"][name]
            full_explanation = interpret_p(p, name)
            st.markdown(f"- **{name.capitalize()}** (observed: {obs:.4f}): {full_explanation}")

        st.caption(
            f"Based on {sig['n_permutations']} permutations at tau={sig['tau']}, "
            f"bins={sig['nbins']}. P-values use the (count + 1) / (N + 1) "
            f"convention to avoid p = 0."
        )
    else:
        st.info("Click **Run permutation test** to assess statistical significance.")

# ── Comparison tab ──
with tab_compare:
    st.subheader("SURD vs Transfer Entropy")
    st.markdown(
        "Transfer entropy (Schreiber, 2000) is the standard information-theoretic "
        "method for measuring causal influence in time series. It produces one number "
        "per source-target pair. It cannot tell you whether two sources carry the "
        "same information or only matter together. SURD can. This tab runs both "
        "methods on the current dataset so you can see the difference."
    )

    te_history = st.slider(
        "Target history length (k)",
        min_value=1, max_value=5, value=1,
        help="How many past steps of the target to condition on. k=1 is standard.")

    if st.button("🆚 Run comparison", type="primary"):
        with st.spinner("Running transfer entropy on each agent..."):
            try:
                comparison = compare_surd_te(
                    df=df, target=target, agents=agents,
                    tau=tau, nbins=nbins, te_history=te_history)
                st.session_state["comparison"] = comparison
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    comparison = st.session_state.get("comparison")
    if comparison:
        # Side-by-side bar chart.
        import plotly.graph_objects as go
        agents_in_compare = [a for a in comparison["transfer_entropy"]
                              if a in comparison["surd_unique"]]
        te_vals = [comparison["transfer_entropy"][a] for a in agents_in_compare]
        u_vals = [comparison["surd_unique"][a] for a in agents_in_compare]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Transfer Entropy", x=agents_in_compare,
                             y=te_vals, marker_color="#888888"))
        fig.add_trace(go.Bar(name="SURD Unique", x=agents_in_compare,
                             y=u_vals, marker_color="#d62828"))
        fig.update_layout(
            title="Per-agent: Transfer Entropy vs SURD Unique",
            yaxis_title="Information (bits)",
            barmode="group", template="plotly_white", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show what SURD reveals that TE misses.
        c1, c2 = st.columns(2)
        with c1:
            st.metric("SURD synergy total",
                       f"{comparison['surd_total_synergy']:.4f} bits",
                       help="Causal info that only appears when agents are combined")
        with c2:
            st.metric("SURD redundancy total",
                       f"{comparison['surd_total_redundant']:.4f} bits",
                       help="Causal info shared across agents")

        st.markdown("### Interpretation")
        st.markdown(explain_comparison(comparison))
    else:
        st.info("Click **Run comparison** to see how SURD compares with transfer entropy on this dataset.")

# ── Explanation tab ──
with tab_explain:
    mode = st.radio("Explanation style:", ["Plain", "Technical"], horizontal=True)
    explanation = generate_explanation(result, mode=mode.lower())
    st.markdown(explanation)

# ── Chat tab ──
with tab_chat:
    chatbot_mode = get_chatbot_mode()
    if chatbot_mode == "ollama":
        st.caption("Using local LLM via Ollama for flexible answers.")
    else:
        st.caption("Using built-in answer engine. Install Ollama for more flexible responses.")

    # Initialise chat history in session state.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history.
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input.
    user_input = st.chat_input("Ask about your results...")
    if user_input:
        # Show user message.
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get answer.
        answer = chat(user_input, result, mode="auto")
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

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
                # Always use the paper's parameters for the benchmark test, not the sidebar's.
                # The benchmarks are designed to verify SURD against known results at tau=1, bins=8.
                eval_results = run_evaluation(tau=1, nbins=8)
                st.session_state["eval_results"] = eval_results
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
    st.caption("Evaluation runs at the paper's parameters (tau=1, bins=8) regardless of sidebar settings.")

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
