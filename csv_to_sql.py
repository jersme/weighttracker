import pandas as pd
import psycopg2
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
        print(f"Error connecting to database: {e}")
        raise

def insert_data_from_csv(csv_file_path):
    """Read data from a CSV file and insert it into the database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Load the CSV file with the correct delimiter
    df = pd.read_csv(csv_file_path, delimiter=';')
    
    for _, row in df.iterrows():
        try:
            # Insert each row into the table and update on conflict
            cursor.execute(
                """
                INSERT INTO weightracker (
                    date, calories_burned, calories_consumed, weight, sports, notes
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    calories_burned = EXCLUDED.calories_burned,
                    calories_consumed = EXCLUDED.calories_consumed,
                    weight = EXCLUDED.weight,
                    sports = EXCLUDED.sports,
                    notes = EXCLUDED.notes
                """,
                (row['date'], row['calories_burned'], row['calories_consumed'], row['Weight'], row['Sports'], row['Notes'])
            )
        except Exception as e:
            print(f"Error inserting data: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Path to the CSV file in the root directory of the repository
    csv_file_path = 'weightracker.csv'
    insert_data_from_csv(csv_file_path)
