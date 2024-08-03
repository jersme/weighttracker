import streamlit as st

# Title of the app
st.title("Weight Tracker")

# Sidebar content
st.sidebar.title("Sidebar Menu")

# Main content
st.header("Main Content Area")

# KPI boxes
kpi1, kpi2, kpi3 = st.columns(3)

# First KPI box
with kpi1:
    st.metric(label="Current Weight", value="70 kg", delta="-1 kg")

# Second KPI box
with kpi2:
    st.metric(label="Target Weight", value="65 kg")

# Third KPI box
with kpi3:
    st.metric(label="Weight Lost", value="5 kg", delta="0.5 kg")

# Additional content below the KPI boxes
st.write("This is where the main content of the app goes.")