import streamlit as st
import psycopg2
import pandas as pd
import os

# Database connection parameters from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

def connect_to_db():
    """Establish a connection to the PostgreSQL database with SSL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            sslmode='require'  # Ensure SSL is used for the connection
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def get_weightracker_data():
    """Fetch data from the weightracker table."""
    conn = connect_to_db()
    if conn is not None:
        try:
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of error
        finally:
            conn.close()

def main():
    """Main function to run the Streamlit app."""
    st.title("Weightracker Data")

    df = get_weightracker_data()

    if not df.empty:
        st.dataframe(df)
    else:
        st.write("No data available.")

if __name__ == "__main__":
    main()
