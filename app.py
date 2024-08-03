import streamlit as st
import pandas as pd
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
    query = "SELECT * FROM weight_data"
    return pd.read_sql(query, engine)

# Function to save new data to the database
def save_data(engine, entry_date, weight, calories_burned, calories_consumed, notes):
    query = text("""
        INSERT INTO weight_data (entry_date, weight, calories_burned, calories_consumed, notes)
        VALUES (:entry_date, :weight, :calories_burned, :calories_consumed, :notes)
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            'entry_date': entry_date,
            'weight': weight,
            'calories_burned': calories_burned,
            'calories_consumed': calories_consumed,
            'notes': notes
        })

# Main app configuration
engine = create_engine_with_ssl()
st.title("Weight Tracker")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Main Dashboard", "Data Entry"])

if page == "Main Dashboard":
    st.header("Weight Data Overview")
    data = load_data(engine)
    if not data.empty:
        st.write(data)
    else:
        st.write("No data available yet.")
elif page == "Data Entry":
    st.header("New Weight Entry")
    with st.form("entry_form", clear_on_submit=True):
        weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
        entry_date = st.date_input("Select the date", value=date.today())
        calories_burned = st.number_input("Enter calories burned", min_value=0, step=1)
        calories_consumed = st.number_input("Enter calories consumed", min_value=0, step=1)
        notes = st.text_area("Additional notes")
        submitted = st.form_submit_button("Submit")

        if submitted:
            save_data(engine, entry_date, weight, calories_burned, calories_consumed, notes)
            st.success("Data saved successfully!")
