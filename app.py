import streamlit as st
import psycopg2
import pandas as pd

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

    # Create two tabs for different sections of the app
    tab1, tab2 = st.tabs(["Analysis", "Data"])

    df = get_weightracker_data()

    with tab1:
        st.header("Analysis")
        if not df.empty:
            # Placeholder for analysis content
            st.subheader("Summary Statistics")
            st.write(df.describe())
            # Add more analysis or visualizations as needed here
        else:
            st.write("No data available for analysis.")

    with tab2:
        st.header("Data")
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
