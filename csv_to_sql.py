import pandas as pd
import psycopg2
import os

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def insert_data_from_csv(csv_file_path):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Load the CSV file with the correct delimiter
    df = pd.read_csv(csv_file_path, delimiter=';')
    
    for _, row in df.iterrows():
        try:
            # Customize this part according to your table structure
            cursor.execute(
                """
                INSERT INTO weightracker (
                    date, calories_burned, calories_consumed, weight, sports, notes
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (row['date'], row['calories_burned'], row['calories_consumed'], row['Weight'], row['Sports'], row['Notes'])
            )
        except Exception as e:
            print(f"Error inserting data: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    insert_data_from_csv('weightracker.csv')  # Specify your CSV file path
