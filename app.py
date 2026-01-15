import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="AadhaarPulse Monitor", layout="wide", page_icon="🆔")

# 2. Function to generate Mock Data (Since you are starting out)
def generate_sample_data():
    dates = pd.date_range(end=datetime.now(), periods=30)
    states = ['Maharashtra', 'Bihar', 'Uttar Pradesh', 'Karnataka', 'Tamil Nadu']
    modalities = ['Fingerprint', 'Iris', 'Face']
    age_groups = ['Child (5-15)', 'Adult (15+)']
    
    data = []
    for date in dates:
        for state in states:
            for age in age_groups:
                for mod in modalities:
                    data.append({
                        'date': date,
                        'state': state,
                        'age_group': age,
                        'modality': mod,
                        'count': np.random.randint(100, 1000)
                    })
    return pd.DataFrame(data)

# 3. Load Data
df = generate_sample_data()

# 4. Sidebar UI (Filters)
st.sidebar.header("📊 Dashboard Filters")
selected_states = st.sidebar.multiselect("Select States", options=df['state'].unique(), default=df['state'].unique()[:2])
selected_modality = st.sidebar.multiselect("Select Modality", options=df['modality'].unique(), default=df['modality'].unique())

# Filter the data based on selection
filtered_df = df[(df['state'].isin(selected_states)) & (df['modality'].isin(selected_modality))]

# 5. Main Dashboard UI
st.title("🆔 Aadhaar Biometric Update Monitor")
st.subheader("Real-time Tracking of Mandatory Biometric Updates (MBU)")

# KPI Metrics
kpi1, kpi2, kpi3 = st.columns(3)
total_updates = filtered_df['count'].sum()
child_updates = filtered_df[filtered_df['age_group'] == 'Child (5-15)']['count'].sum()

kpi1.metric("Total Updates", f"{total_updates:,}")
kpi2.metric("Child Transitions (MBU)", f"{child_updates:,}")
kpi3.metric("Live Sync Status", "Active", delta="Synced")

st.divider()

# 6. Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📈 Update Trends Over Time")
    fig_line = px.line(filtered_df, x='date', y='count', color='modality', title="Daily Updates by Modality")
    st.plotly_chart(fig_line, use_container_width=True)

with col_right:
    st.markdown("### 🎯 Modality Distribution")
    fig_pie = px.pie(filtered_df, values='count', names='modality', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# 7. Age Group Comparison
st.markdown("### 🧒 Children vs 👨 Adults Updates")
fig_bar = px.bar(filtered_df, x='state', y='count', color='age_group', barmode='group')
st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
