import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import date

# Database connection setup
DB_URL = f'postgresql://{st.secrets["DB_USER"]}:{st.secrets["DB_PASSWORD"]}@{st.secrets["DB_HOST"]}:{st.secrets["DB_PORT"]}/{st.secrets["DB_NAME"]}'?sslmode={st.secrets["SSLMODE"]}

engine = create_engine(DB_URL)

# Function to load existing data from the database
def load_data():
    query = "SELECT * FROM weight_data"
    return pd.read_sql(query, engine)

# Function to save new data to the database
def save_data(entry_date, weight, calories_burned, calories_consumed, notes):
    query = f"""
    INSERT INTO weight_data (entry_date, weight, calories_burned, calories_consumed, notes)
    VALUES ('{entry_date}', {weight}, {calories_burned}, {calories_consumed}, '{notes}')
    """
    with engine.connect() as conn:
        conn.execute(query)

# Load existing data
data = load_data()

# Title of the app
st.title("Weight Tracker")

# Sidebar menu
st.sidebar.title("Sidebar Menu")
page = st.sidebar.selectbox("Select a page", ["Main", "Data Entry"])

if page == "Main":
    # Main content
    st.header("Main Content Area")

    # KPI boxes
    if not data.empty:
        current_weight = data["weight"].iloc[-1]
        target_weight = 65  # This can be set dynamically or from user input
        weight_lost = data["weight"].iloc[0] - current_weight

        kpi1, kpi2, kpi3 = st.columns(3)

        # First KPI box
        with kpi1:
            st.metric(label="Current Weight", value=f"{current_weight} kg")

        # Second KPI box
        with kpi2:
            st.metric(label="Target Weight", value=f"{target_weight} kg")

        # Third KPI box
        with kpi3:
            st.metric(label="Weight Lost", value=f"{weight_lost} kg")
    else:
        st.write("No data available.")

    # Display data table
    st.subheader("Weight Data")
    st.write(data)

elif page == "Data Entry":
    # Data Entry page
    st.header("Data Entry")

    # Data entry form
    with st.form("entry_form"):
        weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
        entry_date = st.date_input("Select the date", value=date.today())
        calories_burned = st.number_input("Enter calories burned", min_value=0, step=1)
        calories_consumed = st.number_input("Enter calories consumed", min_value=0, step=1)
        notes = st.text_area("Additional notes")

        submitted = st.form_submit_button("Submit")

        if submitted:
            # Save data to PostgreSQL database
            save_data(entry_date, weight, calories_burned, calories_consumed, notes)

            # Reload data
            data = load_data()

            st.success("Data saved successfully!")
            st.write("Weight:", weight, "kg")
            st.write("Date:", entry_date)
            st.write("Calories Burned:", calories_burned)
            st.write("Calories Consumed:", calories_consumed)
            st.write("Notes:", notes)
