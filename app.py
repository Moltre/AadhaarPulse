import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------- CONFIG ----------

DATA_PATH = "data/biometric_updates.csv"  # put your CSV here


# ---------- DATA LOADING ----------

@st.cache_data(ttl=300)
def load_data(path_or_buffer) -> pd.DataFrame:
    """
    Load Aadhaar demographic/biometric-update-style data and do basic cleaning.
    Expected columns (lowercased):
      - date
      - state
      - district
      - pincode
      - demo_age_5_17
      - demo_age_17_
    """
    df = pd.read_csv(path_or_buffer)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["date", "state", "district", "pincode",
                "demo_age_5_17", "demo_age_17_"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        return pd.DataFrame()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Total “updates” = children + 17+
    df["updates_count"] = (
        pd.to_numeric(df["demo_age_5_17"], errors="coerce").fillna(0)
        + pd.to_numeric(df["demo_age_17_"], errors="coerce").fillna(0)
    )

    # Very simple derived dimensions for filters
    df["age_group"] = "all"      # you can later split rows by age band
    df["modality"] = "all"       # dataset is demographic, keep one category

    return df


# ---------- HELPERS ----------

def filter_data(df: pd.DataFrame,
                state: str,
                district: str,
                date_range) -> pd.DataFrame:
    start_date, end_date = date_range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    if state != "All":
        mask &= df["state"] == state
    if district != "All":
        mask &= df["district"] == district

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

    col1, col2 = st.columns(2)
    col1.metric("Total updates", f"{total_updates:,}")
    col2.metric(
        "Updates today",
        f"{today_updates:,}",
        delta=today_updates - yesterday_updates,
    )


def time_series_chart(df: pd.DataFrame):
    if df.empty:
        return
    daily = df.groupby("date", as_index=False)["updates_count"].sum()
    fig = px.line(
        daily,
        x="date",
        y="updates_count",
        title="Updates over time (all modalities, all ages)",
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
        title="Top 10 states by updates",
    )
    st.plotly_chart(fig, use_container_width=True)


def district_bar_chart(df: pd.DataFrame):
    if df.empty:
        return
    d = (
        df.groupby("district", as_index=False)["updates_count"]
        .sum()
        .sort_values("updates_count", ascending=False)
        .head(10)
    )
    fig = px.bar(
        d,
        x="district",
        y="updates_count",
        title="Top 10 districts by updates",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- STREAMLIT APP ----------

st.set_page_config(
    page_title="Aadhaar Biometric/Demographic Update Monitor",
    layout="wide",
)

st.title("Aadhaar Biometric/Demographic Update Monitor")
st.caption(
    "Dashboard using Aadhaar demographic age-band counts to approximate "
    "biometric update intensity across states and districts."
)


# --- Data source + uploader ---

st.sidebar.header("Data & filters")

if not os.path.exists(DATA_PATH):
    st.sidebar.warning(
        f"Default data file not found: {DATA_PATH}. "
        "Upload a CSV using the uploader below."
    )

uploaded = st.sidebar.file_uploader(
    "Upload latest CSV (optional)",
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
        "Check that the CSV has columns: date, state, district, pincode, "
        "demo_age_5_17, demo_age_17_."
    )
    st.stop()


# --- Sidebar filters ---

min_date = df_raw["date"].min()
max_date = df_raw["date"].max()

state_options = ["All"] + sorted(df_raw["state"].dropna().unique().tolist())
district_options = ["All"] + sorted(df_raw["district"].dropna().unique().tolist())

state = st.sidebar.selectbox("State", state_options)
district = st.sidebar.selectbox("District", district_options)

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
    date_range=(pd.to_datetime(start_date), pd.to_datetime(end_date)),
)


# --- Layout ---

kpi_block(df)

col_left, col_right = st.columns([2, 1])

with col_left:
    time_series_chart(df)
    state_bar_chart(df)

with col_right:
    district_bar_chart(df)

st.subheader("Filtered data (preview)")
st.dataframe(df.head(200))

st.caption(
    "Upload a fresh CSV or change filters to show near real-time changes "
    "in Aadhaar-related update activity for your hackathon demo."
)
