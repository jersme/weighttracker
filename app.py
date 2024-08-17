import streamlit as st
import psycopg2
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Constants
MIN_REQUIRED_POINTS = 5  # Minimum data points required for linear regression to make predictions
CALORIES_PER_KG = 7000  # Caloric equivalent of 1 kg of weight loss
VERSION = "1.0.15"  # Current version of the application

def connect_to_db():
    """
    Establish a connection to the PostgreSQL database using SSL.

    Returns:
        conn (psycopg2.connection): Connection object if successful, None if an error occurs.
    """
    try:
        conn = psycopg2.connect(
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"],
            sslmode='require'  # Enforce SSL for secure connection
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def get_weightracker_data(height_m, target_weight):
    """
    Fetch data from the 'weightracker' table and apply necessary transformations.

    Args:
        height_m (float): User's height in meters.
        target_weight (float): User's target weight in kilograms.

    Returns:
        df (pd.DataFrame): Transformed DataFrame with additional calculated columns.
    """
    conn = connect_to_db()
    if conn is not None:
        try:
            # Fetch all data from the weightracker table
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            
            # Convert the 'date' column to datetime format for time-based calculations
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # Sort the dataframe by date to ensure correct order for cumulative calculations
            df = df.sort_values(by='date')

            # Calculate daily calorie delta (consumed - burned)
            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']

            # Calculate BMI based on user's height
            df['BMI'] = df['weight'] / (height_m ** 2)

            # Calculate kilograms remaining to reach the target weight
            df["kgs_to_target"] = df["weight"] - target_weight

            # Calculate calories needed to save to reach target weight
            df["calories_to_save"] = df["kgs_to_target"] * CALORIES_PER_KG

            # Calculate cumulative calories saved over time
            df['cumulative_calories_saved'] = -df['calorie_delta'].cumsum()  # Inverted to show savings as positive

            # Calculate theoretical kilograms saved based on cumulative calorie savings
            df['theoretical_kgs_saved'] = df['cumulative_calories_saved'] / CALORIES_PER_KG

            # Calculate actual kilograms saved compared to initial weight
            initial_weight = df['weight'].iloc[0]
            df['actual_kgs_saved'] = initial_weight - df['weight']

            # Ensure positive values for theoretical calculations
            df['theoretical_kgs_saved'] = df['theoretical_kgs_saved'].abs()
            df['cumulative_calories_saved'] = df['cumulative_calories_saved'].abs()

            # Calculate daily weight change (delta)
            df['weight_delta'] = df['weight'].diff()

            return df
        except Exception as e:
            st.error(f"Error fetching or transforming data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of an error
        finally:
            conn.close()

def display_metrics(df, target_weight):
    """
    Display key metrics based on the weight tracker data.

    Args:
        df (pd.DataFrame): DataFrame containing weight tracker data.
        target_weight (float): User's target weight in kilograms.
    """
    # Calculate metrics based on the current data
    initial_weight = df['weight'].iloc[0]
    kgs_lost_since_start = df['actual_kgs_saved'].iloc[-1]
    kgs_to_go = df['kgs_to_target'].iloc[-1]
    calories_saved = df['cumulative_calories_saved'].iloc[-1]
    calories_to_go = df['calories_to_save'].iloc[-1]

    # Calculate the total weight to lose and progress percentage
    total_weight_to_lose = initial_weight - target_weight
    progress_percentage = (kgs_lost_since_start / total_weight_to_lose) * 100

    # Display the progress bar in the sidebar
    st.sidebar.write(f"**Progress**: {progress_percentage:.2f}%")
    st.sidebar.progress(progress_percentage / 100)

    # Filter for the current week's data
    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday())
    current_week_data = df[df['date'].dt.date >= start_of_week]

    # Calculate average values for the current week
    avg_kgs_lost_this_week = current_week_data['actual_kgs_saved'].mean() if not current_week_data.empty else 0
    avg_kgs_to_go_this_week = current_week_data['kgs_to_target'].mean() if not current_week_data.empty else 0
    avg_calories_saved_this_week = current_week_data['cumulative_calories_saved'].mean() if not current_week_data.empty else 0
    avg_calories_to_go_this_week = current_week_data['calories_to_save'].mean() if not current_week_data.empty else 0

    # Display the metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kg's Lost Since Start", f"{kgs_lost_since_start:.2f} kg", delta=f"{avg_kgs_lost_this_week:.2f} kg")
    with col2:
        st.metric("Kg's to Go", f"{kgs_to_go:.2f} kg", delta=f"{avg_kgs_to_go_this_week:.2f} kg")
    with col3:
        st.metric("Calories Saved", f"{calories_saved:.0f}", delta=f"{avg_calories_saved_this_week:.0f}")
    with col4:
        st.metric("Calories to Go", f"{calories_to_go:.0f}", delta=f"{avg_calories_to_go_this_week:.0f}")

def main():
    """
    Main function to run the Streamlit app.

    This function initializes the app, fetches and processes data, 
    and displays various analyses and predictions related to weight tracking.
    """
    st.set_page_config(layout="wide")
    st.title("Weight Tracker")

    # Sidebar for additional inputs and information
    st.sidebar.header("Settings")
    target_weight = st.sidebar.number_input("Goal Weight (kg)", min_value=0.0, value=75.0, step=0.1)
    height_m = st.sidebar.number_input("Height (m)", min_value=1.0, value=1.83, step=0.01)

    # Display version information in the sidebar
    st.sidebar.write(f"Version: {VERSION}")

    # Button to refresh data
    if st.sidebar.button("Refresh Data"):
        st.session_state.refresh = True

    # Initialize session state for refreshing data
    if 'refresh' not in st.session_state:
        st.session_state.refresh = True

    # Fetch and process data if the refresh flag is set
    if st.session_state.refresh:
        df = get_weightracker_data(height_m, target_weight)
        st.session_state.df = df
        st.session_state.refresh = False
    else:
        df = st.session_state.df

    # Display metrics, analysis, data, and predictions based on the fetched data
    if not df.empty:
        display_metrics(df, target_weight)
        display_tabs(df, target_weight, height_m)
    else:
        st.write("No data available.")

if __name__ == "__main__":
    main()
