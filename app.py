import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import numpy as np
from pygam import LinearGAM, s

# Function to create database engine
def create_engine_with_ssl():
    # Same database engine creation code as before

# Function to load existing data from the database
def load_data(engine):
    # Same data loading code as before

# Function to save new data to the database
def save_data(engine, entry_date, weight, calories_burned, calories_consumed, notes):
    # Same data saving code as before

# Function to predict the goal weight achievement date using a GAM model
def predict_goal_date(data, goal_weight):
    if len(data) < 5:
        return None, None  # Not enough data to predict

    # Prepare data for modeling
    X = np.arange(len(data))
    y = data['weight'].values

    # Fit a GAM model
    gam = LinearGAM(s(0)).fit(X, y)

    # Predict forward until the goal weight is reached
    for day in range(len(data), len(data) + 365):  # Limit to 1 year
        predicted_weight = gam.predict(np.array([day]))
        if predicted_weight <= goal_weight:
            goal_date = data['entry_date'].iloc[0] + timedelta(days=day)
            days_to_go = (goal_date - datetime.now()).days
            return goal_date.date(), days_to_go
    return None, None

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
        # Existing plotting code

        # Prediction section
        st.header("Predictions")
        predicted_date, days_to_go = predict_goal_date(data, goal_weight)
        if predicted_date:
            col1, col2 = st.columns(2)
            col1.metric("Predicted Date to Reach Goal", predicted_date)
            col2.metric("Days to Go", days_to_go)

            # Add predicted date to the weight trend graph
            fig_weight.add_vline(x=predicted_date, line_dash="dash",
                                 annotation_text="Predicted Goal Date", 
                                 annotation_position="top left")
            st.plotly_chart(fig_weight, use_container_width=True)
        else:
            st.write("Not enough data to predict or goal is beyond 1 year.")

with tab2:
    # Data entry code as before


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
