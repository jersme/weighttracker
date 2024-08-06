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

def get_weightracker_data():
    """Fetch data from the weightracker table and transform it."""
    conn = connect_to_db()
    if conn is not None:
        try:
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            
            # Transform the 'date' column to a datetime format
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # Calculate the daily calorie delta
            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']

            return df
        except Exception as e:
            st.error(f"Error fetching or transforming data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of error
        finally:
            conn.close()

def main():
    """Main function to run the Streamlit app."""
    st.title("Weight Tracker")

    # Sidebar for additional inputs and information
    st.sidebar.header("Settings")
    goal_weight = st.sidebar.number_input("Goal Weight (kg)", min_value=0.0, value=75.0, step=0.1)
    
    df = get_weightracker_data()

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

            AgGrid(summary_df, gridOptions=grid_options)
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

            AgGrid(df, gridOptions=grid_options)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
