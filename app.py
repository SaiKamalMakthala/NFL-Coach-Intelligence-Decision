import sqlite3
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------
# Config
# -----------------------
try:
    from src.config import DB_PATH
except Exception:
    DB_PATH = "nfl_pbp.db"


# -----------------------
# DB helpers
# -----------------------
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


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"


def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{100*x:.1f}%"


# -----------------------
# Football logic (lightweight)
# -----------------------
def compute_field_zone(yardline_100: float) -> str:
    # 0 = opponent endzone, 100 = own endzone (nflfastR convention)
    if yardline_100 <= 10:
        return "Goal-to-go (0-10)"
    if yardline_100 <= 20:
        return "Red zone (11-20)"
    if yardline_100 <= 40:
        return "Plus territory (21-40)"
    if yardline_100 <= 60:
        return "Midfield (41-60)"
    if yardline_100 <= 80:
        return "Own territory (61-80)"
    return "Backed up (81-100)"


def distance_bucket(ydstogo: float) -> str:
    if ydstogo <= 2:
        return "short (0-2)"
    if ydstogo <= 5:
        return "medium (3-5)"
    if ydstogo <= 10:
        return "long (6-10)"
    return "very long (11+)"


def clock_pressure_from_seconds(game_seconds_remaining: float) -> float:
    # normalize: early game -> near 0, late game -> near 1
    # 3600 seconds in regulation
    if game_seconds_remaining is None or np.isnan(game_seconds_remaining):
        return np.nan
    return float(np.clip(1 - (game_seconds_remaining / 3600.0), 0, 1))


def pressure_score_proxy(clock_pressure: float, score_diff_abs: float, yardline_100: float) -> float:
    """
    Proxy score to reflect "leverage":
    - late game (clock_pressure high)
    - close score (score_diff_abs low)
    - field position extreme (very near either end can increase leverage)
    """
    if np.isnan(clock_pressure):
        clock_pressure = 0.5
    if score_diff_abs is None or np.isnan(score_diff_abs):
        score_diff_abs = 7

    # closeness: 0 diff => 1.0, 14+ diff => ~0
    closeness = float(np.clip(1 - (score_diff_abs / 14.0), 0, 1))

    # field leverage: near goal lines => higher
    # yardline_100: 0 is opp EZ, 100 is own EZ
    field_leverage = float(np.clip((abs(yardline_100 - 50) / 50.0), 0, 1))

    # weighted
    return float(np.clip(0.50 * clock_pressure + 0.35 * closeness + 0.15 * field_leverage, 0, 1))


# -----------------------
# Model access (Day 16 output)
# -----------------------
def get_best_feature_table(tables: list[str]) -> str | None:
    for cand in ["feature_table_5", "feature_table_4", "feature_table_3"]:
        if cand in tables:
            return cand
    return None


def load_predictions_if_exist(db_path: str, tables: list[str]) -> pd.DataFrame | None:
    if "fourth_down_predictions" not in tables:
        return None
    return load_table(db_path, "fourth_down_predictions")


def load_recs_if_exist(db_path: str, tables: list[str]) -> pd.DataFrame | None:
    if "fourth_down_recommendations" not in tables:
        return None
    return load_table(db_path, "fourth_down_recommendations")


# -----------------------
# Similar-play retrieval
# -----------------------
def get_similar_plays(df_features: pd.DataFrame, down: int, ydstogo: float, yardline_100: float, n=25):
    """
    Find similar historical plays based on (down, ydstogo, yardline_100).
    Works with your feature tables.
    """
    if df_features is None or len(df_features) == 0:
        return pd.DataFrame()

    if not all(c in df_features.columns for c in ["down", "ydstogo", "yardline_100"]):
        return pd.DataFrame()

    tmp = df_features.copy()
    tmp["ydstogo"] = pd.to_numeric(tmp["ydstogo"], errors="coerce")
    tmp["yardline_100"] = pd.to_numeric(tmp["yardline_100"], errors="coerce")
    tmp["down"] = pd.to_numeric(tmp["down"], errors="coerce")

    tmp = tmp.dropna(subset=["down", "ydstogo", "yardline_100"])
    tmp = tmp[tmp["down"] == down]

    if len(tmp) == 0:
        return pd.DataFrame()

    # distance metric
    tmp["dist"] = (tmp["ydstogo"] - ydstogo).abs() + 0.25 * (tmp["yardline_100"] - yardline_100).abs()
    tmp = tmp.sort_values("dist").head(n)

    return tmp


# -----------------------
# Monte Carlo what-if (from empirical EPA)
# -----------------------
def monte_carlo_epa(df_features: pd.DataFrame, down: int, ydstogo: float, yardline_100: float,
                    play_type: str, n_sims=5000, horizon_plays=1):
    """
    Empirical simulation:
    pick similar plays and sample their EPA as outcome proxy.
    """
    if df_features is None or len(df_features) == 0:
        return None

    needed = {"down", "ydstogo", "yardline_100", "epa", "play_type"}
    if not needed.issubset(df_features.columns):
        return None

    sim_base = df_features.copy()
    sim_base["epa"] = pd.to_numeric(sim_base["epa"], errors="coerce")
    sim_base["ydstogo"] = pd.to_numeric(sim_base["ydstogo"], errors="coerce")
    sim_base["yardline_100"] = pd.to_numeric(sim_base["yardline_100"], errors="coerce")
    sim_base["down"] = pd.to_numeric(sim_base["down"], errors="coerce")

    sim_base["play_type"] = sim_base["play_type"].fillna("other").astype(str).str.lower()

    sim_base = sim_base.dropna(subset=["epa", "ydstogo", "yardline_100", "down"])
    sim_base = sim_base[sim_base["down"] == down]

    # filter by play type if possible
    pt = play_type.lower().strip()
    if pt in ["pass", "run"]:
        sim_base = sim_base[sim_base["play_type"].isin(["pass", "run"])]

    if len(sim_base) < 200:
        # fallback: ignore play_type
        sim_base = df_features.copy()
        sim_base["epa"] = pd.to_numeric(sim_base["epa"], errors="coerce")
        sim_base["ydstogo"] = pd.to_numeric(sim_base["ydstogo"], errors="coerce")
        sim_base["yardline_100"] = pd.to_numeric(sim_base["yardline_100"], errors="coerce")
        sim_base["down"] = pd.to_numeric(sim_base["down"], errors="coerce")
        sim_base = sim_base.dropna(subset=["epa", "ydstogo", "yardline_100", "down"])
        sim_base = sim_base[sim_base["down"] == down]

    if len(sim_base) == 0:
        return None

    # similarity distance
    sim_base["dist"] = (sim_base["ydstogo"] - ydstogo).abs() + 0.25 * (sim_base["yardline_100"] - yardline_100).abs()
    pool = sim_base.sort_values("dist").head(1500)

    if len(pool) < 50:
        return None

    rng = np.random.default_rng(42)
    draws = rng.choice(pool["epa"].values, size=(n_sims, horizon_plays), replace=True)
    totals = draws.sum(axis=1)

    return {
        "n_pool": int(len(pool)),
        "expected": float(np.mean(totals)),
        "p05": float(np.quantile(totals, 0.05)),
        "p50": float(np.quantile(totals, 0.50)),
        "p95": float(np.quantile(totals, 0.95)),
        "prob_negative": float(np.mean(totals < 0)),
        "totals": totals  # for plotting
    }


# -----------------------
# UI: Global Setup
# -----------------------
st.set_page_config(page_title="NFL Coach Decision Intelligence", page_icon="🏈", layout="wide")
st.title("🏈 NFL Coach Decision Intelligence System")
st.caption("Interactive decision support: enter a situation → get recommended actions, risk signals, simulations, and similar-play evidence.")

db_path = str(DB_PATH)
if not Path(db_path).exists():
    st.warning(f"DB not found at {db_path}. Update DB_PATH in src/config.py or place nfl_pbp.db in project root.")

tables = list_tables(db_path)

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Coach Decision Console (Inputs → Outputs)",
            "Analytics Review (What we built)",
            "Raw Table Explorer"
        ],
        index=0,
        key="nav_page"
    )
    st.divider()
    st.caption("DB")
    st.code(db_path, language="text")
    st.caption("Tables")
    st.write(tables)


# =========================
# PAGE 1: Coach Decision Console
# =========================
if page == "Coach Decision Console (Inputs → Outputs)":
    st.subheader("Coach Decision Console")
    st.write("Enter the current game situation. The tool returns recommendation, evidence, and risk-aware what-if outcomes.")

    feature_table = get_best_feature_table(tables)
    df_features = load_table(db_path, feature_table) if feature_table else None

    # ---- Situation Input Form (Coach-friendly) ----
    with st.form("situation_form", clear_on_submit=False):
        st.markdown("### ✅ Situation Inputs")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            down = st.selectbox("Down", [1, 2, 3, 4], index=3, key="form_down")
            ydstogo = st.number_input("Yards to go", min_value=0.0, max_value=40.0, value=2.0, step=0.5, key="form_ydstogo")
            yardline_100 = st.number_input("Yardline_100 (0=Opp EZ, 100=Own EZ)", min_value=0.0, max_value=100.0, value=35.0, step=1.0, key="form_yardline")

        with col2:
            score_diff = st.number_input("Score diff (posteam - defteam)", min_value=-50, max_value=50, value=0, step=1, key="form_scorediff")
            quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=3, key="form_qtr")
            qtr_seconds_remaining = st.number_input("Seconds remaining in quarter", min_value=0, max_value=900, value=420, step=5, key="form_qsecs")

        with col3:
            posteam = st.text_input("Offense team (posteam)", value="KC", key="form_posteam")
            defteam = st.text_input("Defense team (defteam)", value="SF", key="form_defteam")
            play_pref = st.selectbox("Play preference (for simulation)", ["pass", "run", "any"], index=2, key="form_playpref")

        with col4:
            timeout_home = st.number_input("Home timeouts remaining", min_value=0, max_value=3, value=2, step=1, key="form_hto")
            timeout_away = st.number_input("Away timeouts remaining", min_value=0, max_value=3, value=2, step=1, key="form_ato")
            horizon = st.selectbox("Simulation horizon (plays)", [1, 3, 5], index=0, key="form_horizon")

        submitted = st.form_submit_button("Run Decision Support ✅")

    # ---- Compute derived context ----
    if submitted:
        # Derived clock / pressure
        game_seconds_remaining = (4 - quarter) * 900 + qtr_seconds_remaining
        cp = clock_pressure_from_seconds(game_seconds_remaining)
        score_diff_abs = abs(score_diff)
        zone = compute_field_zone(yardline_100)
        distb = distance_bucket(ydstogo)
        ps = pressure_score_proxy(cp, score_diff_abs, yardline_100)

        st.success("Situation processed. See outputs below.")

        # ---- Output header (coach card) ----
        st.markdown("## 📌 Situation Summary")
        a, b, c, d, e = st.columns(5)
        a.metric("Down", f"{down}")
        b.metric("To Go", f"{ydstogo:.1f}")
        c.metric("Yardline_100", f"{yardline_100:.1f}")
        d.metric("Clock Pressure", fmt(cp, 3))
        e.metric("Pressure Score", fmt(ps, 3))

        st.caption(f"Field Zone: **{zone}** | Distance bucket: **{distb}** | Score diff: **{score_diff}**")

        st.divider()

        # ==========================================
        # 1) Recommendation Layer (GO/PUNT/FG)
        # ==========================================
        st.markdown("## 1) ✅ Recommended Action")

        # If you have fourth_down_recommendations table, try to find nearest match
        rec_df = load_recs_if_exist(db_path, tables)
        go_prob = None
        rec_action = None
        evidence_source = None

        if rec_df is not None and all(c in rec_df.columns for c in ["ydstogo", "yardline_100", "recommended_action"]):
            tmp = rec_df.copy()
            tmp["ydstogo"] = pd.to_numeric(tmp["ydstogo"], errors="coerce")
            tmp["yardline_100"] = pd.to_numeric(tmp["yardline_100"], errors="coerce")
            tmp = tmp.dropna(subset=["ydstogo", "yardline_100"])

            # If team columns exist, include them, else situation-only
            if "posteam" in tmp.columns and posteam:
                tmp_team = tmp[tmp["posteam"].astype(str).str.upper() == posteam.upper()]
                if len(tmp_team) > 50:
                    tmp = tmp_team

            tmp["dist"] = (tmp["ydstogo"] - ydstogo).abs() + 0.25 * (tmp["yardline_100"] - yardline_100).abs()
            best = tmp.sort_values("dist").head(1)

            if len(best) == 1:
                row = best.iloc[0]
                rec_action = row.get("recommended_action")
                go_prob = row.get("go_probability", None)
                evidence_source = "fourth_down_recommendations (nearest-match)"

        # Fallback: heuristic if rec table not present
        if rec_action is None:
            # basic coaching heuristic:
            # - short yardage in plus territory => GO
            # - very backed up => PUNT
            # - inside opp 35 and manageable => FG
            if yardline_100 <= 35 and ydstogo <= 5:
                rec_action = "FIELD_GOAL_OR_GO"
            elif yardline_100 <= 55 and ydstogo <= 2:
                rec_action = "GO"
            elif yardline_100 >= 80:
                rec_action = "PUNT"
            else:
                rec_action = "LEAN_PUNT"
            evidence_source = "fallback heuristic (no recommendation table)"

        colR1, colR2, colR3 = st.columns([1, 1, 2])
        colR1.metric("Recommended", str(rec_action))
        colR2.metric("Go Probability", fmt(go_prob, 3) if go_prob is not None else "NA")
        colR3.info(f"Evidence source: {evidence_source}")

        # ==========================================
        # 2) Similar Play Evidence (Day 1–4 + Day 3–4)
        # ==========================================
        st.markdown("## 2) 🔎 Similar-Play Evidence (Historical Context)")

        if df_features is None:
            st.warning("No feature table found (feature_table_3/4/5). Build features first.")
        else:
            sims = get_similar_plays(df_features, down=down, ydstogo=ydstogo, yardline_100=yardline_100, n=40)
            if len(sims) == 0:
                st.warning("Could not find similar plays (missing columns or no data).")
            else:
                # Key metrics from similar plays
                epa_col = "epa" if "epa" in sims.columns else None
                suc_col = "success" if "success" in sims.columns else None

                cA, cB, cC, cD = st.columns(4)
                if epa_col:
                    cA.metric("Similar plays Avg EPA", fmt(pd.to_numeric(sims[epa_col], errors="coerce").mean(), 3))
                if suc_col:
                    cB.metric("Similar plays Success%", pct(pd.to_numeric(sims[suc_col], errors="coerce").mean()))
                cC.metric("Pool size", f"{len(sims):,}")
                cD.metric("Zone", zone)

                show_cols = [c for c in ["game_id", "play_id", "posteam", "defteam", "down", "ydstogo", "yardline_100", "play_type", "epa", "success"] if c in sims.columns]
                st.dataframe(sims[show_cols].head(40), use_container_width=True)

                if epa_col:
                    fig = px.histogram(sims, x=epa_col, nbins=25, title="EPA distribution of similar plays")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ==========================================
        # 3) Monte Carlo What-If (Day 17 + Day 24 style)
        # ==========================================
        st.markdown("## 3) 🎲 What-If Simulator (Monte Carlo EPA)")

        if df_features is None:
            st.warning("Need feature_table with EPA to run simulations.")
        else:
            pt = play_pref if play_pref != "any" else "pass"
            go_sim = monte_carlo_epa(df_features, down, ydstogo, yardline_100, play_type=pt, horizon_plays=horizon)
            punt_sim = monte_carlo_epa(df_features, down, ydstogo + 8, min(100.0, yardline_100 + 15), play_type="run", horizon_plays=horizon)
            fg_sim = monte_carlo_epa(df_features, down, ydstogo, max(0.0, yardline_100 - 10), play_type="pass", horizon_plays=horizon)

            # We treat punt/FG as rough proxies using shifted field position.
            # This is still useful as a portfolio "risk-aware what-if" demonstration.

            sim_rows = []
            for name, sim in [("GO", go_sim), ("PUNT (proxy)", punt_sim), ("FIELD GOAL (proxy)", fg_sim)]:
                if sim is None:
                    continue
                sim_rows.append({
                    "decision": name,
                    "pool_size": sim["n_pool"],
                    "expected_epa": sim["expected"],
                    "p05": sim["p05"],
                    "p50": sim["p50"],
                    "p95": sim["p95"],
                    "prob_negative": sim["prob_negative"]
                })

            if len(sim_rows) == 0:
                st.warning("Simulation failed (not enough comparable EPA samples).")
            else:
                sim_df = pd.DataFrame(sim_rows).sort_values("expected_epa", ascending=False)
                st.dataframe(sim_df, use_container_width=True)

                fig = px.bar(sim_df, x="decision", y="expected_epa", title="Expected EPA by Decision (simulation)")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.scatter(sim_df, x="expected_epa", y="prob_negative", color="decision",
                                  title="Value vs Risk (Prob Negative EPA)")
                st.plotly_chart(fig2, use_container_width=True)

                # If user wants deeper view: show GO totals distribution
                with st.expander("Show distribution (GO)"):
                    if go_sim is not None:
                        dist_df = pd.DataFrame({"total_epa": go_sim["totals"]})
                        fig3 = px.histogram(dist_df, x="total_epa", nbins=40, title="GO outcome distribution (simulated)")
                        st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ==========================================
        # 4) Timeout Alert Guidance (Day 23 style)
        # ==========================================
        st.markdown("## 4) ⏱️ Timeout Guidance (Alert)")

        # rule-based alert:
        # late game, close game, offense in plus territory, timeouts available
        alert = (cp >= 0.75) and (score_diff_abs <= 7) and (yardline_100 <= 60)
        msg = "Consider timeout: late/close/critical field position." if alert else "No urgent timeout signal in this state."

        colT1, colT2, colT3 = st.columns(3)
        colT1.metric("Alert", "YES" if alert else "NO")
        colT2.metric("Home TO", int(timeout_home))
        colT3.metric("Away TO", int(timeout_away))
        st.info(msg)

        st.divider()

        # ==========================================
        # 5) Narrative Summary (Day 25 style)
        # ==========================================
        st.markdown("## 5) 📝 Coach-Ready Narrative (Auto Summary)")

        narrative = [
            f"Situation: **{down}th & {ydstogo:.1f}** at yardline_100 **{yardline_100:.1f}** ({zone}).",
            f"Game context: quarter **{quarter}**, **{qtr_seconds_remaining}** seconds left in quarter (clock pressure={fmt(cp,3)}), score diff={score_diff}.",
            f"Pressure proxy: **{fmt(ps,3)}** (higher means higher leverage).",
            f"Recommendation: **{rec_action}**" + (f" (Go prob={fmt(go_prob,3)})" if go_prob is not None else "") + ".",
            "Evidence: Similar historical situations were retrieved to support the recommendation.",
            "Risk: Monte Carlo simulation shows expected EPA and downside risk by decision option."
        ]
        for line in narrative:
            st.write("•", line)


# =========================
# PAGE 2: Analytics Review
# =========================
elif page == "Analytics Review (What we built)":
    st.subheader("Analytics Review")
    st.write("This page is for analysts / interviewers: tables, KPIs, and quick visuals across your pipeline.")

    feature_table = get_best_feature_table(tables)
    if feature_table:
        df = load_table(db_path, feature_table)

        st.markdown(f"### Feature table: `{feature_table}`")
        st.caption("Shows the engineered feature backbone (days 3–15).")

        # quick KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")

        if "epa" in df.columns:
            c2.metric("Avg EPA", fmt(pd.to_numeric(df["epa"], errors="coerce").mean(), 3))
        if "success" in df.columns:
            c3.metric("Success%", pct(pd.to_numeric(df["success"], errors="coerce").mean()))
        if "pressure_score" in df.columns:
            c4.metric("Avg Pressure", fmt(pd.to_numeric(df["pressure_score"], errors="coerce").mean(), 3))

        # distributions
        if "ydstogo" in df.columns:
            fig = px.histogram(df, x="ydstogo", nbins=40, title="Yards-to-go distribution")
            st.plotly_chart(fig, use_container_width=True)
        if "yardline_100" in df.columns:
            fig = px.histogram(df, x="yardline_100", nbins=40, title="Field position distribution (yardline_100)")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.head(200), use_container_width=True)
    else:
        st.warning("No feature_table found (feature_table_3/4/5).")

    st.divider()

    # 4th down artifacts
    if "fourth_down_predictions" in tables:
        pred = load_table(db_path, "fourth_down_predictions")
        st.markdown("### `fourth_down_predictions` (Day 16)")
        st.dataframe(pred.head(200), use_container_width=True)

        if "go_probability" in pred.columns:
            fig = px.histogram(pred, x="go_probability", nbins=30, title="Go probability distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `fourth_down_predictions` found yet.")

    if "fourth_down_feature_importance" in tables:
        imp = load_table(db_path, "fourth_down_feature_importance").sort_values("importance_mean", ascending=False)
        st.markdown("### `fourth_down_feature_importance` (Day 20)")
        fig = px.bar(imp, x="feature", y="importance_mean", error_y="importance_std", title="Feature importance")
        st.plotly_chart(fig, use_container_width=True)

    if "endgame_simulation_summary" in tables:
        st.markdown("### `endgame_simulation_summary` (Day 24)")
        endg = load_table(db_path, "endgame_simulation_summary")
        st.dataframe(endg, use_container_width=True)

    if "narrative_summaries" in tables:
        st.markdown("### `narrative_summaries` (Day 25)")
        ns = load_table(db_path, "narrative_summaries")
        st.dataframe(ns.head(100), use_container_width=True)


# =========================
# PAGE 3: Raw Table Explorer
# =========================
elif page == "Raw Table Explorer":
    st.subheader("Raw Table Explorer")
    if not tables:
        st.warning("No tables found in DB.")
    else:
        chosen = st.selectbox("Select a table", tables, key="explorer_table_select")
        limit = st.slider("Row limit", min_value=50, max_value=5000, value=500, step=50, key="explorer_limit")
        df = load_table(db_path, chosen, limit=limit)
        st.dataframe(df, use_container_width=True)

        with st.expander("Column stats"):
            st.write(df.describe(include="all").transpose())
