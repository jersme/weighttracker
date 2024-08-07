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
    
    # Ensure the date is treated as a string
    df['date'] = df['date'].astype(str)
    
    for _, row in df.iterrows():
        try:
            # Print the row being processed for debugging
            print(f"Processing row: {row.to_dict()}")
            
            # First, try to update the existing record with new values
            cursor.execute(
                """
                UPDATE weightracker SET
                    calories_burned = %s,
                    calories_consumed = %s,
                    weight = %s,
                    sports = %s,
                    notes = %s
                WHERE date = %s
                """,
                (row['calories_burned'], row['calories_consumed'], row['Weight'], row['Sports'], row['Notes'], row['date'])
            )
            
            # If no rows were affected by the update, it means the record doesn't exist, so insert a new one
            if cursor.rowcount == 0:
                print(f"Inserting new row for date {row['date']}")
                cursor.execute(
                    """
                    INSERT INTO weightracker (
                        date, calories_burned, calories_consumed, weight, sports, notes
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (row['date'], row['calories_burned'], row['calories_consumed'], row['Weight'], row['Sports'], row['Notes'])
                )
            else:
                print(f"Updated existing row for date {row['date']}")
                
        except Exception as e:
            print(f"Error inserting/updating data for date {row['date']}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()

def print_table_contents():
    """Retrieve and print the contents of the weightracker table."""
    conn = connect_to_db()
    cursor = conn.cursor()

    try:
        # Execute a query to fetch all data from the weightracker table
        cursor.execute("SELECT * FROM weightracker ORDER BY date;")
        rows = cursor.fetchall()

        # Print the column headers
        colnames = [desc[0] for desc in cursor.description]
        print("\t".join(colnames))

        # Print each row
        for row in rows:
            print("\t".join(map(str, row)))

    except Exception as e:
        print(f"Error fetching data: {e}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Path to the CSV file in the root directory of the repository
    csv_file_path = 'weightracker.csv'
    insert_data_from_csv(csv_file_path)
    print_table_contents()  # Print the final contents of the table
