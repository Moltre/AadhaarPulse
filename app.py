import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------- CONFIG ----------

DATA_PATH = "data/biometric_updates.csv"  # default local file

# Change this to your real count column name from the CSV (lowercase):
# e.g. "count", "biometricupdate", "no_of_updates"
ORIGINAL_COUNT_COL = "count"


# ---------- DATA LOADING ----------

@st.cache_data(ttl=300)
def load_data(path_or_buffer) -> pd.DataFrame:
    """
    Load Aadhaar biometric update data and do basic cleaning.
    Works for both local file path and uploaded file.
    """
    df = pd.read_csv(path_or_buffer)

    # Standardize column names (lowercase, no extra spaces)
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure the chosen count column exists
    if ORIGINAL_COUNT_COL.lower() not in df.columns:
        st.error(
            f"Configured count column '{ORIGINAL_COUNT_COL}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
        return pd.DataFrame()

    # Expect: date, state, district, age_group, modality, <count>
    if "date" not in df.columns:
        st.error("CSV must contain a 'date' column.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Create unified 'updates_count' column from your original column
    df["updates_count"] = pd.to_numeric(
        df[ORIGINAL_COUNT_COL.lower()], errors="coerce"
    ).fillna(0)

    # Fill missing dimension columns if absent
    for col in ["state", "district", "age_group", "modality"]:
        if col not in df.columns:
            df[col] = "Unknown"

    return df


# ---------- HELPERS ----------

def filter_data(df: pd.DataFrame,
                state: str,
                district: str,
                age_group: str,
                modality: str,
                date_range) -> pd.DataFrame:
    start_date, end_date = date_range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    if state != "All":
        mask &= df["state"] == state
    if district != "All":
        mask &= df["district"] == district
    if age_group != "All":
        mask &= df["age_group"] == age_group
    if modality != "All":
        mask &= df["modality"] == modality

    return df[mask]


def kpi_block(df: pd.DataFrame):
    if df.empty:
        st.warning("No data for selected filters.")
        return

    total_updates = int(df["updates_count"].sum())

    today = df["date"].max()
    yesterday = today - timedelta(days=1)

    today_updates = int(df.loc[df["date"] == today, "updates_count"].sum())
    yesterday_updates = int(df.loc[df["date"] == yesterday, "updates_count"].sum())

    # Simple children vs adults heuristic
    children_mask = df["age_group"].astype(str).str.contains(
        "0-5|5-15|child", case=False, regex=True
    )
    children_updates = int(df.loc[children_mask, "updates_count"].sum())
    adult_updates = total_updates - children_updates

    col1, col2, col3 = st.columns(3)
    col1.metric("Total updates", f"{total_updates:,}")
    col2.metric(
        "Updates today",
        f"{today_updates:,}",
        delta=today_updates - yesterday_updates,
    )
    col3.metric(
        "Children vs adults",
        f"{children_updates:,} / {adult_updates:,}",
        help="Children (0‑15) vs adults (others)",
    )


def time_series_chart(df: pd.DataFrame):
    if df.empty:
        return
    daily = df.groupby(["date", "modality"], as_index=False)["updates_count"].sum()
    fig = px.line(
        daily,
        x="date",
        y="updates_count",
        color="modality",
        title="Biometric updates over time by modality",
    )
    st.plotly_chart(fig, use_container_width=True)


def state_bar_chart(df: pd.DataFrame):
    if df.empty:
        return
    s = (
        df.groupby("state", as_index=False)["updates_count"]
        .sum()
        .sort_values("updates_count", ascending=False)
        .head(10)
    )
    fig = px.bar(
        s,
        x="state",
        y="updates_count",
        title="Top 10 states by biometric updates",
    )
    st.plotly_chart(fig, use_container_width=True)


def modality_pie_chart(df: pd.DataFrame):
    if df.empty:
        return
    m = df.groupby("modality", as_index=False)["updates_count"].sum()
    fig = px.pie(
        m,
        names="modality",
        values="updates_count",
        title="Share of updates by biometric modality",
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- STREAMLIT APP ----------

st.set_page_config(
    page_title="Aadhaar Biometric Update Monitor",
    layout="wide",
)

st.title("Aadhaar Biometric Update Monitor")
st.caption(
    "Real‑time style dashboard of Aadhaar biometric updates "
    "(fingerprint, iris, face) across states, districts, and age groups."
)

# --- Data source + uploader ---

st.sidebar.header("Data & filters")

if not os.path.exists(DATA_PATH):
    st.sidebar.warning(
        f"Default data file not found: {DATA_PATH}. "
        "You can upload a CSV using the uploader below."
    )

uploaded = st.sidebar.file_uploader(
    "Upload latest biometric update CSV (optional)",
    type=["csv"],
    help="Upload a fresh export to instantly refresh charts.",
)

if uploaded is not None:
    df_raw = load_data(uploaded)
elif os.path.exists(DATA_PATH):
    df_raw = load_data(DATA_PATH)
else:
    df_raw = pd.DataFrame()

if df_raw.empty:
    st.error(
        "Dataset is empty or could not be loaded. "
        "Check your CSV and ORIGINAL_COUNT_COL in app.py."
    )
    st.stop()

# --- Sidebar filters ---

min_date = df_raw["date"].min()
max_date = df_raw["date"].max()

state_options = ["All"] + sorted(df_raw["state"].dropna().unique().tolist())
district_options = ["All"] + sorted(df_raw["district"].dropna().unique().tolist())
age_group_options = ["All"] + sorted(df_raw["age_group"].dropna().unique().tolist())
modality_options = ["All"] + sorted(df_raw["modality"].dropna().unique().tolist())

state = st.sidebar.selectbox("State", state_options)
district = st.sidebar.selectbox("District", district_options)
age_group = st.sidebar.selectbox("Age group", age_group_options)
modality = st.sidebar.selectbox("Biometric modality", modality_options)

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

df = filter_data(
    df_raw,
    state=state,
    district=district,
    age_group=age_group,
    modality=modality,
    date_range=(pd.to_datetime(start_date), pd.to_datetime(end_date)),
)

# --- Layout ---

kpi_block(df)

col_left, col_right = st.columns([2, 1])

with col_left:
    time_series_chart(df)
    state_bar_chart(df)

with col_right:
    modality_pie_chart(df)

st.subheader("Filtered data (preview)")
st.dataframe(df.head(200))

st.caption(
    "Tip: Upload a fresh CSV or change filters to show near real-time changes "
    "in Aadhaar biometric updates during your hackathon demo."
)
