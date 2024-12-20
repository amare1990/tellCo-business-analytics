import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class user_overview_analyzer:
    def __init__(self):
        # Initialize database connection parameters from environment variables
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")

    def load_data_from_postgres(self, query):
        """
        Connects to the PostgreSQL database and loads data based on the provided SQL query.

        :param query: SQL query to execute.
        :return: DataFrame containing the results of the query.
        """
        try:
            # Establish a connection to the database
            connection = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )

            # Load data using pandas
            df = pd.read_sql_query(query, connection)

            # Close the database connection
            connection.close()

            return df

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
