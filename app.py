"""Lumen — automated data analysis dashboard.

Upload a file (or load the sample) and Lumen instantly builds a dashboard:
headline numbers, plain-English findings, auto-chosen charts, and a "records to
review" anomaly check. Power tools (Explore, Clean, SQL, Predict, Report) live in
the tabs across the top — no setup required to get the first insights.

Run:  ./run.sh        (or)   env -u PYTHONPATH ./venv/bin/python -m streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src import (analysis, anomaly, cleaning, data_loader, db, insights,
                 modeling, profiling, report, state, viz)
from src.sample_data import make_customers

st.set_page_config(page_title="Lumen", page_icon="💡", layout="wide")


# ── Cached heavy lifting (re-runs only when the data changes) ─────────────────
@st.cache_data(show_spinner=False)
def _anomaly(df, sensitivity):
    return anomaly.detect_anomalies(df, sensitivity)


# ── Header + data loading ────────────────────────────────────────────────────
st.title("💡 Lumen")
st.caption("Upload your data and get an instant dashboard — no spreadsheets, no code.")

with st.container(border=True):
    c1, c2 = st.columns([3, 2])
    with c1:
        uploaded = st.file_uploader(
            "Drop a data file here (Excel, CSV, JSON, Parquet)",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
            label_visibility="visible",
        )
        if uploaded is not None and st.button("Analyze this file", type="primary"):
            try:
                df = data_loader.coerce_datetimes(data_loader.load_file(uploaded))
                state.set_data(df, name=uploaded.name.rsplit(".", 1)[0])
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't read that file: {e}")
    with c2:
        st.markdown("**No file handy?**")
        st.write("Load a sample customer dataset to see Lumen in action.")
        if st.button("Try the sample dataset"):
            state.set_data(make_customers(), name="sample_customers")
            st.rerun()

df = state.get_df()
if df is None:
    st.info("👆 Upload a file or load the sample to build your dashboard.")
    st.stop()

st.success(f"Analyzing **{st.session_state.get('dataset_name', 'your data')}** — "
           f"{df.shape[0]:,} records, {df.shape[1]} fields.")

tab_dash, tab_explore, tab_clean, tab_sql, tab_predict, tab_report = st.tabs(
    ["📊 Dashboard", "🔎 Explore", "🧹 Clean up", "🗄️ Ask in SQL", "🔮 Predict", "📄 Report"]
)

# ── DASHBOARD: auto-everything ───────────────────────────────────────────────
with tab_dash:
    cards = insights.kpi_cards(df)
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.metric(card["label"], card["value"])

    st.subheader("📝 Key findings")
    for f in insights.key_findings(df):
        st.markdown(f"- {f}")

    st.subheader("🚩 Records to review")
    st.caption("Lumen flags rows that look unusual compared to the rest — a first pass "
               "for fraud, errors, or anything worth a closer look.")
    sensitivity = st.slider("How strict should the check be? (higher = flags more)",
                            1, 15, 3, help="Approx. % of records to flag as unusual.") / 100
    try:
        scored, flagged, summary = _anomaly(df, sensitivity)
        a1, a2 = st.columns([1, 3])
        a1.metric("Flagged", f"{summary['flagged']:,}", f"{summary['flagged_pct']}% of records")
        with a2:
            for d in summary["drivers"]:
                st.markdown(f"- {d}")
        if not flagged.empty:
            st.dataframe(flagged.head(100), use_container_width=True)
            st.download_button("⬇️ Download flagged records (CSV)",
                               flagged.to_csv(index=False), file_name="records_to_review.csv")
        st.session_state["anomaly_summary"] = summary
    except Exception as e:  # noqa: BLE001
        st.info(f"Anomaly check needs at least a couple of numeric/categorical fields. ({e})")

    st.subheader("📈 Automatic charts")
    auto = insights.auto_charts(df)
    st.session_state["auto_charts"] = auto
    grid = st.columns(2)
    for i, (title, fig) in enumerate(auto):
        with grid[i % 2]:
            st.markdown(f"**{title}**")
            st.plotly_chart(fig, use_container_width=True, key=f"auto_{i}")

# ── EXPLORE: simple build-your-own chart ─────────────────────────────────────
with tab_explore:
    st.subheader("Build your own chart")
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    kind = st.selectbox("What kind of chart?",
                        ["Bar (counts)", "Histogram (spread)", "Scatter (relationship)",
                         "Line (trend)", "Box (compare groups)", "Pie (share)"])
    fig = None
    try:
        if kind.startswith("Bar"):
            x = st.selectbox("Category", cat_cols or all_cols)
            fig = viz.bar(df, x)
        elif kind.startswith("Histogram"):
            x = st.selectbox("Number", num_cols or all_cols)
            fig = viz.histogram(df, x)
        elif kind.startswith("Scatter"):
            x = st.selectbox("Horizontal", num_cols)
            y = st.selectbox("Vertical", num_cols, index=min(1, len(num_cols) - 1))
            color = st.selectbox("Color by (optional)", [None] + all_cols)
            fig = viz.scatter(df, x, y, color)
        elif kind.startswith("Line"):
            x = st.selectbox("Horizontal (often a date)", all_cols)
            y = st.selectbox("Vertical (a number)", num_cols)
            fig = viz.line(df, x, y)
        elif kind.startswith("Box"):
            y = st.selectbox("Number", num_cols)
            x = st.selectbox("Group by (optional)", [None] + cat_cols)
            fig = viz.box(df, y, x)
        else:
            names = st.selectbox("Category", cat_cols or all_cols)
            fig = viz.pie(df, names)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't build that chart: {e}")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    if len(num_cols) >= 2:
        st.subheader("What moves together")
        st.dataframe(analysis.top_correlations(df), use_container_width=True)

# ── CLEAN UP: the common fixes, in plain language ────────────────────────────
with tab_clean:
    st.subheader("Tidy up your data")

    def _apply(result):
        new_df, message, code = result
        state.update_df(new_df)
        state.log_step(message, code)
        st.success(message)
        st.rerun()

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Remove duplicate rows"):
            _apply(cleaning.drop_duplicates(df))
        miss_cols = st.multiselect("Fill in blanks for…", df.columns.tolist())
        how = st.selectbox("Fill with", ["typical value (median)", "average (mean)",
                                          "most common", "remove those rows"])
        strat = {"typical value (median)": "median", "average (mean)": "mean",
                 "most common": "mode", "remove those rows": "drop_rows"}[how]
        if st.button("Fill blanks") and miss_cols:
            _apply(cleaning.impute_missing(df, miss_cols, strat))
    with cc2:
        drop = st.multiselect("Remove columns I don't need", df.columns.tolist())
        if st.button("Remove columns") and drop:
            _apply(cleaning.drop_columns(df, drop))
        out_cols = st.multiselect("Tame extreme values in…",
                                  df.select_dtypes(include="number").columns.tolist())
        if st.button("Cap extreme values") and out_cols:
            _apply(cleaning.handle_outliers(df, out_cols, "iqr", "clip"))

    if st.button("↩️ Undo everything (reset to original)"):
        state.reset_df()
        st.rerun()

    recipe = state.get_recipe()
    if recipe:
        st.markdown(f"**Steps applied ({len(recipe)}):**")
        for i, s in enumerate(recipe, 1):
            st.write(f"{i}. {s['message']}")
        st.download_button("⬇️ Save these steps as a Python script",
                           state.recipe_to_script(), file_name="cleaning_pipeline.py")
    st.dataframe(state.get_df().head(50), use_container_width=True)

# ── ASK IN SQL ───────────────────────────────────────────────────────────────
with tab_sql:
    st.subheader("Query your data with SQL")
    st.caption("Your data is a table called **data**. Write any SELECT query.")
    with st.expander("See the columns"):
        st.dataframe(db.schema_preview(df), use_container_width=True)
    query = st.text_area("SQL", "SELECT *\nFROM data\nLIMIT 100;", height=150)
    if st.button("Run query", type="primary"):
        try:
            st.session_state["sql_result"] = db.run_query(df, query)
        except Exception as e:  # noqa: BLE001
            st.error(f"Query error: {e}")
    res = st.session_state.get("sql_result")
    if res is not None:
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download results (CSV)", res.to_csv(index=False),
                           file_name="query_result.csv")

# ── PREDICT (AutoML, plain language) ─────────────────────────────────────────
with tab_predict:
    st.subheader("Predict an outcome")
    st.caption("Pick the column you'd like to predict; Lumen trains several models and "
               "tells you how well they do and which factors matter most.")
    target = st.selectbox("What do you want to predict?", df.columns.tolist())
    auto_task = modeling.detect_task(df, target)
    nice = "a category (yes/no, type, group)" if auto_task == "classification" else "a number"
    st.write(f"Looks like you're predicting **{nice}**.")
    if st.button("Train models", type="primary"):
        with st.spinner("Training models…"):
            try:
                results, leaderboard = modeling.train_and_compare(df, target, auto_task)
                st.session_state["model_results"] = results
                st.session_state["model_leaderboard"] = leaderboard
                st.session_state["model_task"] = auto_task
                st.session_state["model_best_name"] = results[0].name
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't train models: {e}")

    results = st.session_state.get("model_results")
    if results:
        lb = st.session_state["model_leaderboard"]
        task = st.session_state["model_task"]
        best = results[0]
        if task == "classification":
            st.metric("Best model accuracy", f"{best.metrics['accuracy'] * 100:.0f}%",
                      help=f"{best.name}")
        else:
            st.metric("Best model R² (fit quality)", f"{best.metrics['r2']:.2f}", help=f"{best.name}")
        st.markdown(f"Best model: **{best.name}**")
        st.dataframe(lb, use_container_width=True)
        if best.importance is not None:
            st.markdown("**What matters most for the prediction**")
            st.plotly_chart(viz.feature_importance_fig(best.importance), use_container_width=True)
        if task == "classification":
            cm, labels = modeling.confusion(best)
            st.plotly_chart(viz.confusion_matrix_fig(cm, labels), use_container_width=True)
        else:
            st.plotly_chart(viz.residuals_fig(best.y_test, best.y_pred), use_container_width=True)

# ── REPORT ───────────────────────────────────────────────────────────────────
with tab_report:
    st.subheader("Export a shareable report")
    name = st.session_state.get("dataset_name", "dataset")
    title = st.text_input("Report title", f"Data Report — {name}")
    st.write("Includes the dashboard's headline numbers, key findings, anomaly summary, "
             "automatic charts, and any prediction results.")
    if st.button("Generate report", type="primary"):
        figures = st.session_state.get("auto_charts", [])
        html = report.build_report(
            title=title,
            cards=insights.kpi_cards(df),
            profile_df=profiling.column_profile(df),
            findings=insights.key_findings(df),
            anomaly=st.session_state.get("anomaly_summary"),
            quality_flags=profiling.quality_flags(df),
            cleaning_steps=[s["message"] for s in state.get_recipe()],
            figures=figures,
            leaderboard_df=st.session_state.get("model_leaderboard"),
            best_model=st.session_state.get("model_best_name"),
            task=st.session_state.get("model_task"),
        )
        st.session_state["report_html"] = html
        st.success("Report ready — download below, then open it or print to PDF.")
    html = st.session_state.get("report_html")
    if html:
        st.download_button("⬇️ Download report (HTML)", html,
                           file_name=f"{name}_report.html", mime="text/html")
