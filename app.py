import streamlit as st
import pandas as pd
import joblib
import os

# --- Page setup ---
st.set_page_config(page_title="Cricket Player Analyzer", layout="wide")
st.title("🏏 IPL Player Performance Analyzer")

# --- Load data and model ---
@st.cache_data
def load_data():
    model_path = "player_fit_model.pkl"
    data_path = "player_profiles_master_scored_clean.csv"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        st.error("❌ Missing data or model file. Please ensure both files are in the same folder as app.py.")
        st.stop()

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
    features = model_bundle.get("features", []) if isinstance(model_bundle, dict) else []

    df = pd.read_csv(data_path)

    # Clean invalid player names
    # No longer fill with 'Unknown_Player' or cast to str, keep actual names from cleaned dataset

    df = df.fillna(0)

    if features:
        X = df[features]
        df["fit_prediction"] = model.predict(X)
    else:
        st.warning("⚠️ Feature list missing in model file. Predictions may be inaccurate.")
        df["fit_prediction"] = 0

    return df


df = load_data()

# --- 🧠 Helper Functions for Roles, XI, and Comparison ---
def classify_role(row):
    """Return a simple role label using available columns."""
    try:
        runs = float(row.get("total_runs", 0))
        wickets = float(row.get("total_wickets", 0))
    except Exception:
        runs = 0
        wickets = 0

    if wickets >= 5 and runs >= 200:
        return "All-rounder"
    if wickets >= 5:
        return "Bowler"
    return "Batter"


def build_simple_xi(df, top_n=11):
    """
    Build a balanced XI:
    - 5 top batters by overall_score
    - 1-2 all-rounders by combined score (runs + wickets)
    - 4 top bowlers by wickets_per_match
    - Fill remaining with best overall_score players
    """
    if "batter" not in df.columns:
        return []

    # Ensure necessary columns exist
    for col in ["overall_score", "wickets_per_match", "total_wickets", "total_runs"]:
        if col not in df.columns:
            df[col] = 0

    # --- Filter by role ---
    batters = df[df["role"] == "Batter"].sort_values("overall_score", ascending=False)
    bowlers = df[df["role"] == "Bowler"].sort_values("wickets_per_match", ascending=False)
    allrounders = df[df["role"] == "All-rounder"].sort_values(
        ["total_wickets", "total_runs"], ascending=False
    )

    # --- Select players ---
    selected = []

    # Pick top 5 batters
    selected.extend(list(batters["batter"].dropna().astype(str).unique())[:5])

    # Pick top 1-2 all-rounders
    selected.extend(list(allrounders["batter"].dropna().astype(str).unique())[:2])

    # Pick top 4 bowlers
    selected.extend(
        [p for p in list(bowlers["batter"].dropna().astype(str).unique()) if p not in selected][:4]
    )

    # If still not enough, fill with remaining best players
    if len(selected) < top_n:
        extras = list(df.sort_values("overall_score", ascending=False)["batter"].dropna().astype(str).unique())
        for name in extras:
            if name not in selected:
                selected.append(name)
            if len(selected) >= top_n:
                break

    return selected[:top_n]


def compare_players(df, names):
    """
    Compare selected players by stats side by side.
    """
    cols = [
        "batter",
        "overall_score",
        "batting_average",
        "strike_rate",
        "total_runs",
        "total_wickets",
        "wickets_per_match",
        "economy",
    ]
    available = [c for c in cols if c in df.columns]
    sub = df[df["batter"].isin(names)][available].copy()
    if "batter" in sub.columns:
        sub = sub.set_index("batter")
    return sub
# --- End of Helper Section ---

# --- Sidebar IPL Teams ---

# --- Sidebar filters ---
st.sidebar.header("🔍 Filters")
match_col = "matches_played_x" if "matches_played_x" in df.columns else "matches_played"
min_matches = st.sidebar.slider("Minimum Matches Played", 0, int(df[match_col].max()), 10)
batting_type = st.sidebar.selectbox("Select Batting Style", ["All"] + sorted(df["batting_style"].unique()))
bowling_type = st.sidebar.selectbox("Select Bowling Style", ["All"] + sorted(df["bowling_style"].unique()))

filtered_df = df[
    (df[match_col] >= min_matches)
    & ((df["batting_style"] == batting_type) | (batting_type == "All"))
    & ((df["bowling_style"] == bowling_type) | (bowling_type == "All"))
]

st.sidebar.write(f"Showing {len(filtered_df)} players")

# --- Main display tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Player Stats", "💪 Fit Player Insights", "🧠 Model Info", "🧩 Role & XI", "🔎 Compare Players"])

with tab1:
    st.subheader("Player Performance Summary")
    st.dataframe(
        filtered_df[
            [
                "batter",
                match_col,
                "batting_style",
                "bowling_style",
                "batting_average",
                "strike_rate",
                "total_wickets",
                "economy",
                "overall_score",
            ]
        ].sort_values("overall_score", ascending=False),
        use_container_width=True,
    )

with tab2:
    st.subheader("Top Performing Players")
    top_players = filtered_df.sort_values("overall_score", ascending=False).head(15)
    st.bar_chart(top_players.set_index("batter")["overall_score"])

    st.subheader("Fit Player Breakdown")
    fit_counts = df["fit_prediction"].value_counts()
    st.write(f"✅ Fit Players: {fit_counts.get(1, 0)} | ❌ Others: {fit_counts.get(0, 0)}")

with tab3:
    st.subheader("Model and Feature Overview")
    st.markdown("**Model Used:** Random Forest Classifier")
    if "fit_prediction" in df.columns:
        st.write("**Predictions Generated Successfully ✅**")
    else:
        st.warning("Model did not generate predictions.")
    st.write("**Feature Columns Used:**")
    # Use the features variable safely, fallback if not available
    features = []
    try:
        # Try to get features from the filtered_df's columns that are not obvious outputs
        # However, prefer to use the features variable from load_data if possible
        import inspect
        frame = inspect.currentframe()
        while frame:
            if "features" in frame.f_locals:
                features = frame.f_locals["features"]
                break
            frame = frame.f_back
    except Exception:
        features = []
    st.code(", ".join(features) if features else "No features available")

# --- 🧩 Role Classification & XI Tab ---
with tab4:
    st.subheader("Role classification & Suggested XI")
    if "role" not in df.columns:
        df["role"] = df.apply(classify_role, axis=1)

    st.write("Sample role distribution:")
    st.dataframe(df.groupby("role").size().rename("count").reset_index(), use_container_width=True)

    st.write("---")
    st.subheader("Suggested Playing XI (simple heuristic)")
    top_n = st.slider("XI size", 5, 15, 11)
    xi = build_simple_xi(df, top_n=top_n)
    if xi:
        st.write(f"Suggested XI ({len(xi)}):")
        for i, name in enumerate(xi, start=1):
            st.markdown(f"{i}. **{name}**")
    else:
        st.info("Not enough data to suggest an XI.")

# --- 🔎 Player Comparison Tab ---
with tab5:
    st.subheader("Compare players")
    all_names = list(df["batter"].dropna().astype(str).unique()) if "batter" in df.columns else []
    selected = st.multiselect("Select up to 4 players to compare", all_names, max_selections=4)
    if selected:
        comp = compare_players(df, selected)
        st.dataframe(comp, use_container_width=True)
    else:
        st.info("Choose players from the dropdown to compare their stats.")