import streamlit as st
import pandas as pd
from datetime import date

# Function to load existing data or create a new DataFrame if the file does not exist
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Weight", "Calories Burned", "Calories Consumed", "Notes"])

# Function to save data to a CSV file
def save_data(file_path, data):
    data.to_csv(file_path, index=False)

# Load existing data
data_file = "weight_data.csv"
data = load_data(data_file)

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
        current_weight = data["Weight"].iloc[-1]
        target_weight = 65  # This can be set dynamically or from user input
        weight_lost = data["Weight"].iloc[0] - current_weight

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
            # Add new entry to data
            new_entry = {
                "Date": entry_date,
                "Weight": weight,
                "Calories Burned": calories_burned,
                "Calories Consumed": calories_consumed,
                "Notes": notes
            }
            data = data.append(new_entry, ignore_index=True)

            # Save updated data
            save_data(data_file, data)

            st.success("Data saved successfully!")
            st.write("Weight:", weight, "kg")
            st.write("Date:", entry_date)
            st.write("Calories Burned:", calories_burned)
            st.write("Calories Consumed:", calories_consumed)
            st.write("Notes:", notes)