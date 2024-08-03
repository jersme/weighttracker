import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import date

# Function to create database engine
def create_engine_with_ssl():
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_NAME = st.secrets["DB_NAME"]
    SSLMODE = st.secrets["SSLMODE"]
    DB_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={SSLMODE}'
    return create_engine(DB_URL)

# Function to load existing data from the database
def load_data(engine):
    query = "SELECT * FROM weight_data ORDER BY entry_date"
    return pd.read_sql(query, engine)

# Function to save new data to the database
def save_data(engine, entry_date, weight, calories_burned, calories_consumed, notes):
    query = text("""
        INSERT INTO weight_data (entry_date, weight, calories_burned, calories_consumed, notes)
        VALUES (:entry_date, :weight, :calories_burned, :calories_consumed, :notes)
    """)
    try:
        with engine.connect() as conn:
            conn.execute(query, {
                'entry_date': entry_date,
                'weight': weight,
                'calories_burned': calories_burned,
                'calories_consumed': calories_consumed,
                'notes': notes
            })
        st.success("Data saved successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False
    return True

# Main app configuration
engine = create_engine_with_ssl()
st.title("Weight Tracker")

# Sidebar for goal weight input
st.sidebar.title("Settings")
goal_weight = st.sidebar.number_input("Goal Weight (kg)", value=78, min_value=0)

tab1, tab2 = st.tabs(["Main Dashboard", "Data Entry"])

with tab1:
    st.header("Weight Data Overview")
    data = load_data(engine)
    if not data.empty:
        current_weight = data['weight'].iloc[-1]
        kg_lost = data['weight'].iloc[0] - current_weight
        kg_to_go = goal_weight - current_weight

        col1, col2, col3 = st.columns(3)
        col1.metric("Kg's to go", f"{kg_to_go:.2f} kg")
        col2.metric("Kg's lost", f"{kg_lost:.2f} kg")
        col3.metric("Current weight", f"{current_weight:.2f} kg")

        # Plotly chart
        fig = px.line(data, x='entry_date', y='weight', title='Weight Tracking Over Time',
                      markers=True, labels={'weight': 'Weight (kg)', 'entry_date': 'Date'})
        fig.add_hline(y=goal_weight, line_dash="dot",
                      annotation_text="Goal Weight", 
                      annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available yet.")

with tab2:
    st.header("New Weight Entry")
    with st.form("entry_form"):
        weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
        entry_date = st.date_input("Select the date", value=date.today())
        calories_burned = st.number_input("Enter calories burned", min_value=0, step=1)
        calories_consumed = st.number_input("Enter calories consumed", min_value=0, step=1)
        notes = st.text_area("Additional notes")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if save_data(engine, entry_date, weight, calories_burned, calories_consumed, notes):
                st.balloons()  # Celebrate successful entry
