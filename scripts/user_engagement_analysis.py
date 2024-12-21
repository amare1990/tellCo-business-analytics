import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from dotenv import load_dotenv

class UserEngagementAnalysis:
    def __init__(self):
        # Initialize database connection parameters from environment variables
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")


    def load_data_from_postgres(self, query):
        """
        Connects to the PostgreSQL database using SQLAlchemy and loads data based on the provided SQL query.

        :param query: SQL query to execute.
        :return: DataFrame containing the results of the query.
        """
        try:
            # Create a SQLAlchemy engine
            engine = create_engine(
                f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            )

            # Load data using pandas
            with engine.connect() as connection:
                df = pd.read_sql_query(query, connection)

            return df

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

