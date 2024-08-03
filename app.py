import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime

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
    query = "SELECT entry_date, weight FROM weight_data ORDER BY entry_date"
    df = pd.read_sql(query, engine)
    df['entry_date'] = pd.to_datetime(df['entry_date'])  # Ensure 'entry_date' is a datetime type
    return df

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
        data['kg_to_go'] = data['weight'] - goal_weight  # Calculate kg's to go as the difference from the goal

        # Plot for weight over time
        fig_weight = px.line(data, x='entry_date', y='weight', title='Weight Tracking Over Time',
                             markers=True, labels={'weight': 'Weight (kg)', 'entry_date': 'Date'})
        fig_weight.add_hline(y=goal_weight, line_dash="dot",
                             annotation_text="Goal Weight", 
                             annotation_position="bottom right")
        st.plotly_chart(fig_weight, use_container_width=True)

        # Plot for kg's to go over time, zero-centered at the goal weight
        fig_kgs_to_go = px.line(data, x='entry_date', y='kg_to_go', title="Kg's to Go Over Time",
                                markers=True, labels={'kg_to_go': "Kg's to Go", 'entry_date': 'Date'})
        fig_kgs_to_go.update_layout(yaxis_title="Kg's Above/Below Goal")
        st.plotly_chart(fig_kgs_to_go, use_container_width=True)

    else:
        st.write("No data available yet.")

with tab2:
    st.header("New Weight Entry")
    with st.form("entry_form"):
        weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
        entry_date = st.date_input("Select the date", value=datetime.now())
        calories_burned = st.number_input("Enter calories burned", min_value=0, step=1)
        calories_consumed = st.number_input("Enter calories consumed", min_value=0, step=1)
        notes = st.text_area("Additional notes")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if save_data(engine, entry_date, weight, calories_burned, calories_consumed, notes):
                st.balloons()  # Celebrate successful entry
