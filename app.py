import streamlit as st
import psycopg2
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

def connect_to_db():
    """Establish a connection to the PostgreSQL database with SSL."""
    try:
        conn = psycopg2.connect(
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"],
            sslmode='require'  # Ensure SSL is used for the connection
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def get_weightracker_data(height_m, target_weight):
    """Fetch data from the weightracker table and transform it."""
    conn = connect_to_db()
    if conn is not None:
        try:
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            
            # Transform the 'date' column to a datetime format
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # Sort the dataframe by date to correctly calculate cumulative values
            df = df.sort_values(by='date')

            # Calculate the daily calorie delta
            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']

            # Calculate BMI using the user's height
            df['BMI'] = df['weight'] / (height_m ** 2)

            # Calculate KGs to go to target weight
            df["kgs_to_target"] = df["weight"] - target_weight

            # Calculate calories to save to reach target weight
            df["calories_to_save"] = df["kgs_to_target"] * 7000

            # Calculate cumulative calories saved
            df['cumulative_calories_saved'] = df['calorie_delta'].cumsum()

            return df
        except Exception as e:
            st.error(f"Error fetching or transforming data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of error
        finally:
            conn.close()

def main():
    """Main function to run the Streamlit app."""
    # Use Streamlit's wide mode to use more horizontal space
    st.set_page_config(layout="wide")
    st.title("Weight Tracker")

    # Sidebar for additional inputs and information
    st.sidebar.header("Settings")
    target_weight = st.sidebar.number_input("Goal Weight (kg)", min_value=0.0, value=75.0, step=0.1)
    height_m = st.sidebar.number_input("Height (m)", min_value=1.0, value=1.83, step=0.01)  # Allow decimal values for height

    # Button to refresh data
    if st.sidebar.button("Refresh Data"):
        st.session_state.refresh = True

    # Initialize session state for refreshing
    if 'refresh' not in st.session_state:
        st.session_state.refresh = True

    if st.session_state.refresh:
        df = get_weightracker_data(height_m, target_weight)
        st.session_state.df = df
        st.session_state.refresh = False
    else:
        df = st.session_state.df

    # Create two tabs for different sections of the app
    tab1, tab2 = st.tabs(["Analysis", "Data"])

    with tab1:
        st.header("Analysis")
        if not df.empty:
            st.subheader("Summary Statistics")
            # Use AG Grid for displaying summary statistics
            summary_df = df.describe().reset_index()  # Reset index for better display
            gb = GridOptionsBuilder.from_dataframe(summary_df)
            gb.configure_pagination(paginationAutoPageSize=True)  # Pagination
            gb.configure_side_bar()  # Enable side bar for filtering and more
            grid_options = gb.build()

            # Use AG Grid with full width
            AgGrid(summary_df, gridOptions=grid_options, height=300, width='100%')
            # Add more analysis or visualizations as needed here
        else:
            st.write("No data available for analysis.")

    with tab2:
        st.header("Data")
        if not df.empty:
            # Use AG Grid for displaying the main data
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            grid_options = gb.build()

            # Use AG Grid with full width
            AgGrid(df, gridOptions=grid_options, height=400, width='100%')
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
