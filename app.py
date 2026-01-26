import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Project DB path ---
try:
    from src.config import DB_PATH
except Exception:
    DB_PATH = "nfl_pbp.db"


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def list_tables(db_path: str):
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    conn.close()
    return tables


@st.cache_data(show_spinner=False)
def load_table(db_path: str, table: str, limit: int | None = None):
    conn = sqlite3.connect(db_path)
    q = f"SELECT * FROM {table}"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def safe_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{100*x:.1f}%"


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"


def add_common_filters(df: pd.DataFrame):
    """
    Apply filters from sidebar based on columns that exist.
    Returns filtered df.
    """
    # Core columns
    posteam = safe_col(df, ["posteam", "team", "offense_team"])
    defteam = safe_col(df, ["defteam", "defense_team", "opp_team"])
    down = safe_col(df, ["down"])
    ydstogo = safe_col(df, ["ydstogo", "yards_to_go", "to_go"])
    yardline_100 = safe_col(df, ["yardline_100", "yardline", "yardline100"])
    risk_bucket = safe_col(df, ["risk_bucket"])
    field_zone = safe_col(df, ["field_zone"])

    # Sidebar filters (only show if column exists)
    with st.sidebar:
        st.subheader("Filters")

        if posteam:
            teams = sorted([t for t in df[posteam].dropna().unique().tolist() if str(t).strip() != ""])
            selected_team = st.selectbox("Offense Team (posteam)", ["All"] + teams, index=0)
        else:
            selected_team = "All"

        if defteam:
            dteams = sorted([t for t in df[defteam].dropna().unique().tolist() if str(t).strip() != ""])
            selected_def = st.selectbox("Defense Team (defteam)", ["All"] + dteams, index=0)
        else:
            selected_def = "All"

        if down:
            downs = sorted([int(x) for x in pd.to_numeric(df[down], errors="coerce").dropna().unique().tolist()])
            selected_down = st.multiselect("Down", downs, default=downs)
        else:
            selected_down = None

        if ydstogo:
            y = pd.to_numeric(df[ydstogo], errors="coerce")
            y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
            selected_yd = st.slider("Yards To Go", min_value=float(y_min), max_value=float(y_max), value=(float(y_min), float(y_max)))
        else:
            selected_yd = None

        if yardline_100:
            yl = pd.to_numeric(df[yardline_100], errors="coerce")
            yl_min, yl_max = float(np.nanmin(yl)), float(np.nanmax(yl))
            selected_yardline = st.slider("Yardline_100 (0=EZ, 100=own goal)", min_value=float(yl_min), max_value=float(yl_max), value=(float(yl_min), float(yl_max)))
        else:
            selected_yardline = None

        if field_zone:
            zones = sorted([z for z in df[field_zone].dropna().unique().tolist()])
            selected_zone = st.multiselect("Field Zone", zones, default=zones)
        else:
            selected_zone = None

        if risk_bucket:
            buckets = sorted([b for b in df[risk_bucket].dropna().unique().tolist()])
            selected_risk = st.multiselect("Risk Bucket", buckets, default=buckets)
        else:
            selected_risk = None

    # Apply filters
    out = df.copy()

    if posteam and selected_team != "All":
        out = out[out[posteam] == selected_team]

    if defteam and selected_def != "All":
        out = out[out[defteam] == selected_def]

    if down and selected_down is not None and len(selected_down) > 0:
        out = out[pd.to_numeric(out[down], errors="coerce").isin(selected_down)]

    if ydstogo and selected_yd is not None:
        y = pd.to_numeric(out[ydstogo], errors="coerce")
        out = out[(y >= selected_yd[0]) & (y <= selected_yd[1])]

    if yardline_100 and selected_yardline is not None:
        yl = pd.to_numeric(out[yardline_100], errors="coerce")
        out = out[(yl >= selected_yardline[0]) & (yl <= selected_yardline[1])]

    if field_zone and selected_zone is not None and len(selected_zone) > 0:
        out = out[out[field_zone].isin(selected_zone)]

    if risk_bucket and selected_risk is not None and len(selected_risk) > 0:
        out = out[out[risk_bucket].isin(selected_risk)]

    return out


# =========================
# App Setup
# =========================
st.set_page_config(page_title="NFL Coach Decision Intelligence", page_icon="🏈", layout="wide")
st.title("🏈 NFL Coach Decision Intelligence System")
st.caption("Front-office style decision support: features → models → simulations → coach outputs")

db_path = str(DB_PATH)
if not Path(db_path).exists():
    st.warning(f"DB not found at {db_path}. Update DB_PATH in src/config.py or place nfl_pbp.db in project root.")

tables = list_tables(db_path)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Executive Overview",
            "Data Foundations (Day 1–4)",
            "Feature Engineering (Day 3–15)",
            "4th Down Decisioning (Day 16–20)",
            "Simulations (Day 17 & 24)",
            "Coach Report (Day 22–25)",
            "Data Quality (Day 13)",
            "Raw Table Explorer"
        ],
        index=0
    )
    st.divider()
    st.caption("Connected DB")
    st.code(db_path, language="text")
    st.caption("Tables")
    st.write(tables)


# =========================
# Page: Executive Overview
# =========================
if page == "Executive Overview":
    st.subheader("Executive Overview")

    # Choose best “current” table for context
    feature_table = None
    for cand in ["feature_table_5", "feature_table_4", "feature_table_3"]:
        if cand in tables:
            feature_table = cand
            break

    rec_table = "fourth_down_recommendations" if "fourth_down_recommendations" in tables else None
    alerts_table = "timeout_alerts" if "timeout_alerts" in tables else None

    c1, c2, c3, c4 = st.columns(4)

    if feature_table:
        df_feat = load_table(db_path, feature_table)
        df_feat_f = add_common_filters(df_feat)

        with c1:
            kpi_card("Rows (filtered)", f"{len(df_feat_f):,}", f"From {feature_table}")
        with c2:
            epa_col = safe_col(df_feat_f, ["epa"])
            if epa_col:
                kpi_card("Avg EPA", fmt(pd.to_numeric(df_feat_f[epa_col], errors="coerce").mean(), 3))
            else:
                kpi_card("Avg EPA", "NA")
        with c3:
            succ_col = safe_col(df_feat_f, ["success"])
            if succ_col:
                kpi_card("Success Rate", pct(pd.to_numeric(df_feat_f[succ_col], errors="coerce").mean()))
            else:
                kpi_card("Success Rate", "NA")
        with c4:
            pr_col = safe_col(df_feat_f, ["pressure_score"])
            if pr_col:
                kpi_card("Avg Pressure", fmt(pd.to_numeric(df_feat_f[pr_col], errors="coerce").mean(), 3))
            else:
                kpi_card("Avg Pressure", "NA")
    else:
        st.info("No feature_table found yet (feature_table_5/4/3).")

    st.divider()

    # Recommendations visuals
    if rec_table:
        recs = load_table(db_path, rec_table)
        recs_f = add_common_filters(recs)

        st.markdown("### 4th Down Recommendations")
        colA, colB = st.columns([1, 1])

        with colA:
            if "recommended_action" in recs_f.columns:
                mix = recs_f["recommended_action"].value_counts().reset_index()
                mix.columns = ["action", "count"]
                fig = px.pie(mix, names="action", values="count", title="Recommendation Mix (filtered)")
                st.plotly_chart(fig, use_container_width=True)

        with colB:
            gp = safe_col(recs_f, ["go_probability"])
            if gp:
                fig = px.histogram(recs_f, x=gp, nbins=30, title="Go Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top GO Candidates")
        show_cols = [c for c in ["game_id", "play_id", "posteam", "defteam", "ydstogo", "yardline_100", "go_probability", "recommended_action"] if c in recs_f.columns]
        if "go_probability" in recs_f.columns:
            st.dataframe(recs_f.sort_values("go_probability", ascending=False)[show_cols].head(25), use_container_width=True)
        else:
            st.dataframe(recs_f[show_cols].head(25), use_container_width=True)
    else:
        st.info("No `fourth_down_recommendations` table yet (Day 18).")

    # Alerts
    if alerts_table:
        st.markdown("### Timeout Alerts")
        alerts = load_table(db_path, alerts_table)
        alerts_f = add_common_filters(alerts)
        st.dataframe(alerts_f.head(200), use_container_width=True)
    else:
        st.info("No `timeout_alerts` table yet (Day 23).")


# =========================
# Page: Data Foundations
# =========================
elif page == "Data Foundations (Day 1–4)":
    st.subheader("Data Foundations (Day 1–4)")
    st.write("Use this section to show the pipeline foundation: ingestion → cleaned → core distributions.")

    # Try to locate a “raw-ish” table or feature table for distributions
    candidate = None
    for cand in ["feature_table_3", "feature_table_4", "feature_table_5"]:
        if cand in tables:
            candidate = cand
            break

    if not candidate:
        st.warning("No feature tables found to visualize. Build at least up to Day 3/4.")
    else:
        df = load_table(db_path, candidate)
        df = add_common_filters(df)

        st.markdown(f"### Using table: `{candidate}`")
        st.caption("Shows distributions that prove data coverage and baseline validity.")

        # Missingness overview
        with st.expander("Missingness Overview"):
            miss = df.isna().mean().sort_values(ascending=False).reset_index()
            miss.columns = ["column", "missing_rate"]
            fig = px.bar(miss.head(30), x="column", y="missing_rate", title="Top Missing Columns (first 30)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss.head(50), use_container_width=True)

        # Down/Distance/Field
        c1, c2 = st.columns(2)

        down_col = safe_col(df, ["down"])
        ydstogo_col = safe_col(df, ["ydstogo", "yards_to_go", "to_go"])
        yardline_col = safe_col(df, ["yardline_100", "yardline", "yardline100"])

        with c1:
            if down_col:
                fig = px.histogram(df, x=down_col, nbins=4, title="Down Distribution")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if ydstogo_col:
                fig = px.histogram(df, x=ydstogo_col, nbins=40, title="Yards-To-Go Distribution")
                st.plotly_chart(fig, use_container_width=True)

        if yardline_col:
            fig = px.histogram(df, x=yardline_col, nbins=40, title="Field Position (yardline_100) Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Quick table sample
        st.markdown("### Sample Rows")
        st.dataframe(df.head(50), use_container_width=True)


# =========================
# Page: Feature Engineering
# =========================
elif page == "Feature Engineering (Day 3–15)":
    st.subheader("Feature Engineering (Day 3–15)")
    st.write("This is where you show your football understanding: down-distance, field zones, pressure, risk.")

    # Choose best table
    feature_table = None
    for cand in ["feature_table_5", "feature_table_4", "feature_table_3"]:
        if cand in tables:
            feature_table = cand
            break

    if not feature_table:
        st.warning("No feature tables found. Build through Day 15 first.")
    else:
        df = load_table(db_path, feature_table)
        df = add_common_filters(df)

        st.markdown(f"### Using `{feature_table}`")

        # Visual: EPA vs distance bucket / field zone if present
        epa = safe_col(df, ["epa"])
        dist_bucket = safe_col(df, ["distance_bucket"])
        field_zone = safe_col(df, ["field_zone"])
        pressure = safe_col(df, ["pressure_score"])
        risk = safe_col(df, ["decision_risk_score"])

        c1, c2 = st.columns(2)

        with c1:
            if dist_bucket and epa:
                agg = df.groupby(dist_bucket)[epa].mean().reset_index()
                fig = px.bar(agg, x=dist_bucket, y=epa, title="Avg EPA by Distance Bucket")
                st.plotly_chart(fig, use_container_width=True)
            elif epa:
                fig = px.histogram(df, x=epa, nbins=60, title="EPA Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if field_zone and epa:
                agg = df.groupby(field_zone)[epa].mean().reset_index()
                fig = px.bar(agg, x=field_zone, y=epa, title="Avg EPA by Field Zone")
                st.plotly_chart(fig, use_container_width=True)
            elif pressure:
                fig = px.histogram(df, x=pressure, nbins=50, title="Pressure Score Distribution")
                st.plotly_chart(fig, use_container_width=True)

        # Risk/pressure relationship
        if risk and pressure and epa:
            tmp = df[[risk, pressure, epa]].copy()
            tmp[risk] = pd.to_numeric(tmp[risk], errors="coerce")
            tmp[pressure] = pd.to_numeric(tmp[pressure], errors="coerce")
            tmp[epa] = pd.to_numeric(tmp[epa], errors="coerce")
            tmp = tmp.dropna()
            fig = px.scatter(tmp.sample(min(len(tmp), 5000)), x=pressure, y=epa, color=risk,
                             title="EPA vs Pressure (colored by Decision Risk)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Table Preview")
        st.dataframe(df.head(200), use_container_width=True)


# =========================
# Page: 4th Down Decisioning
# =========================
elif page == "4th Down Decisioning (Day 16–20)":
    st.subheader("4th Down Decisioning (Day 16–20)")
    st.write("Model outputs, validation slices, and explainability (what drives aggressiveness).")

    pred_table = "fourth_down_predictions" if "fourth_down_predictions" in tables else None
    slice_table = "fourth_down_validation_slices" if "fourth_down_validation_slices" in tables else None
    imp_table = "fourth_down_feature_importance" if "fourth_down_feature_importance" in tables else None

    if not pred_table:
        st.warning("No `fourth_down_predictions` table found. Run Day 16 first.")
    else:
        df = load_table(db_path, pred_table)
        df = add_common_filters(df)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Pred rows (filtered)", f"{len(df):,}")
        with c2:
            if "actual_go" in df.columns:
                kpi_card("True GO rate", pct(df["actual_go"].astype(float).mean()))
        with c3:
            if "go_probability" in df.columns:
                kpi_card("Avg GO prob", fmt(df["go_probability"].astype(float).mean(), 3))
        with c4:
            if "predicted_go" in df.columns:
                kpi_card("Pred GO rate", pct(df["predicted_go"].astype(float).mean()))

        # Distribution & calibration-ish view
        if "go_probability" in df.columns:
            fig = px.histogram(df, x="go_probability", nbins=30, title="Go Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Predictions Table")
        st.dataframe(df.head(200), use_container_width=True)

    if slice_table:
        st.markdown("### Bias / Slice Report (Day 19)")
        slices = load_table(db_path, slice_table)
        st.dataframe(slices, use_container_width=True)

        # Heatmap-style
        if set(["ydstogo_bucket", "field_bucket", "avg_go_prob"]).issubset(slices.columns):
            pivot = slices.pivot(index="ydstogo_bucket", columns="field_bucket", values="avg_go_prob")
            fig = px.imshow(pivot, text_auto=True, title="Avg GO Prob by Distance x Field Bucket")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No slice table yet (`fourth_down_validation_slices`). Run Day 19 after rebuilding predictions with features.")

    if imp_table:
        st.markdown("### Explainability (Day 20)")
        imp = load_table(db_path, imp_table).sort_values("importance_mean", ascending=False)
        fig = px.bar(imp, x="feature", y="importance_mean", error_y="importance_std",
                     title="Permutation Importance (ROC-AUC drop)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp, use_container_width=True)
    else:
        st.info("No feature importance table yet (`fourth_down_feature_importance`). Run Day 20 after predictions include feature columns.")


# =========================
# Page: Simulations
# =========================
elif page == "Simulations (Day 17 & 24)":
    st.subheader("Simulations (Day 17 & 24)")
    st.write("Monte Carlo what-if outcomes and endgame horizon outcomes.")

    if "monte_carlo_whatif_results" in tables:
        sims = load_table(db_path, "monte_carlo_whatif_results")
        st.markdown("### Monte Carlo What-If (Day 17)")
        st.dataframe(sims, use_container_width=True)

        if set(["strategy", "expected_epa"]).issubset(sims.columns):
            fig = px.bar(sims, x="strategy", y="expected_epa",
                         title="Expected EPA by Strategy (What-If)")
            st.plotly_chart(fig, use_container_width=True)

            # Risk view if quantiles exist
            if set(["p05_epa", "p95_epa"]).issubset(sims.columns):
                fig = px.scatter(sims, x="expected_epa", y="prob_negative_epa", color="strategy",
                                 title="Value vs Risk (Prob Negative EPA)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `monte_carlo_whatif_results` table yet (Day 17).")

    st.divider()

    if "endgame_simulation_summary" in tables:
        endg = load_table(db_path, "endgame_simulation_summary")
        st.markdown("### Endgame Simulation Summary (Day 24)")
        st.dataframe(endg, use_container_width=True)

        # Visualize quantiles if present
        cols = ["expected_total_epa", "p05_total_epa", "p50_total_epa", "p95_total_epa"]
        if all(c in endg.columns for c in cols):
            one = endg.iloc[0]
            chart_df = pd.DataFrame({
                "metric": ["p05", "p50", "expected", "p95"],
                "total_epa": [one["p05_total_epa"], one["p50_total_epa"], one["expected_total_epa"], one["p95_total_epa"]]
            })
            fig = px.bar(chart_df, x="metric", y="total_epa", title="Endgame Total EPA Distribution Summary")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `endgame_simulation_summary` table yet (Day 24). Run Day 24.")


# =========================
# Page: Coach Report
# =========================
elif page == "Coach Report (Day 22–25)":
    st.subheader("Coach Report (Day 22–25)")
    st.write("Coach-facing insights, alerts, and narrative summaries.")

    if "coach_insights" in tables:
        st.markdown("### Coach Insights (Day 22)")
        ci = load_table(db_path, "coach_insights")
        for line in ci["insight"].dropna().tolist():
            st.write("•", line)
    else:
        st.info("No `coach_insights` table yet (Day 22).")

    st.divider()

    if "narrative_summaries" in tables:
        st.markdown("### Narrative Summaries (Day 25)")
        ns = load_table(db_path, "narrative_summaries")
        for line in ns["summary_line"].dropna().tolist():
            st.write(line)
    else:
        st.info("No `narrative_summaries` table yet (Day 25).")

    st.divider()

    if "timeout_alerts" in tables:
        st.markdown("### Timeout Alerts (Day 23)")
        alerts = load_table(db_path, "timeout_alerts")
        alerts = add_common_filters(alerts)
        st.dataframe(alerts.head(300), use_container_width=True)
    else:
        st.info("No `timeout_alerts` table yet (Day 23).")


# =========================
# Page: Data Quality
# =========================
elif page == "Data Quality (Day 13)":
    st.subheader("Data Quality (Day 13)")
    if "data_quality_report" not in tables:
        st.warning("No `data_quality_report` found. Run Day 13 sanity checks.")
    else:
        dq = load_table(db_path, "data_quality_report")
        st.dataframe(dq, use_container_width=True)

        fails = dq[dq["status"] == "FAIL"] if "status" in dq.columns else pd.DataFrame()
        if len(fails) > 0:
            st.error(f"❌ Failures found: {len(fails)}")
            st.dataframe(fails, use_container_width=True)
        else:
            st.success("✅ No failures in data_quality_report")


# =========================
# Page: Raw Table Explorer
# =========================
elif page == "Raw Table Explorer":
    st.subheader("Raw Table Explorer")
    if not tables:
        st.warning("No tables found in DB.")
    else:
        chosen = st.selectbox("Select a table", tables)
        limit = st.slider("Row limit", min_value=50, max_value=5000, value=500, step=50)
        df = load_table(db_path, chosen, limit=limit)
        df = add_common_filters(df)
        st.dataframe(df, use_container_width=True)

        with st.expander("Column stats"):
            st.write(df.describe(include="all").transpose())
