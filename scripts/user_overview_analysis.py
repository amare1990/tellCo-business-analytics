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

    def user_overview_analysis(self):
        """
        Performs user overview analysis to answer the following:
        - Top 10 handsets
        - Top 3 handset manufacturers
        - Top 5 handsets per top 3 manufacturers
        """
        # Dictionary to store results
        analysis_results = {}

        # Query for Top 10 handsets
        query_top_10_handsets = """
        SELECT "Handset Type" AS handset, COUNT(*) AS usage_count
        FROM xdr_data
        GROUP BY "Handset Type"
        ORDER BY usage_count DESC
        LIMIT 10
        """
        top_10_handsets = self.load_data_from_postgres(query_top_10_handsets)
        analysis_results["top_10_handsets"] = top_10_handsets

        # Query for Top 3 handset manufacturers
        query_top_3_manufacturers = """
        SELECT "Handset Manufacturer" AS manufacturer, COUNT(*) AS usage_count
        FROM xdr_data
        GROUP BY "Handset Manufacturer"
        ORDER BY usage_count DESC
        LIMIT 3
        """
        top_3_manufacturers = self.load_data_from_postgres(query_top_3_manufacturers)
        analysis_results["top_3_manufacturers"] = top_3_manufacturers

        # Query for Top 5 handsets per top 3 manufacturers
        if not top_3_manufacturers.empty:
            manufacturer_list = top_3_manufacturers["manufacturer"].tolist()
            manufacturer_filter = "', '".join(manufacturer_list)
            query_top_5_per_manufacturer = f"""
            WITH RankedHandsets AS (
                SELECT
                    "Handset Manufacturer" AS manufacturer,
                    "Handset Type" AS handset,
                    COUNT(*) AS usage_count,
                    ROW_NUMBER() OVER (PARTITION BY "Handset Manufacturer" ORDER BY COUNT(*) DESC) AS rank
                FROM xdr_data
                WHERE "Handset Manufacturer" IN ('{manufacturer_filter}')
                GROUP BY "Handset Manufacturer", "Handset Type"
            )
            SELECT manufacturer, handset, usage_count
            FROM RankedHandsets
            WHERE rank <= 5
            ORDER BY manufacturer, rank;
            """
            top_5_per_manufacturer = self.load_data_from_postgres(query_top_5_per_manufacturer)
            analysis_results["top_5_per_manufacturer"] = top_5_per_manufacturer
        else:
            analysis_results["top_5_per_manufacturer"] = pd.DataFrame()


        return analysis_results

    def user_behavior_analysis(self):
        """
        Aggregates user behavior data:
        - Number of xDR sessions
        - Session duration (in seconds)
        - Total download (DL) and upload (UL) data (in Bytes)
        - Total data volume (in Bytes) during each session for each application
        """
        query = """
            SELECT
                "IMSI" AS user_id,
                CASE
                    WHEN "Social Media DL (Bytes)" > 0 OR "Social Media UL (Bytes)" > 0 THEN 'Social Media'
                    WHEN "Gaming DL (Bytes)" > 0 OR "Gaming UL (Bytes)" > 0 THEN 'Gaming'
                    WHEN "YouTube DL (Bytes)" > 0 OR "YouTube UL (Bytes)" > 0 THEN 'YouTube'
                    WHEN "Netflix DL (Bytes)" > 0 OR "Netflix UL (Bytes)" > 0 THEN 'Netflix'
                    WHEN "Google DL (Bytes)" > 0 OR "Google UL (Bytes)" > 0 THEN 'Google'
                    ELSE 'Other'
                END AS application,
                COUNT("bearer id") AS session_count,
                SUM("Dur. (s)") AS total_duration,
                SUM("Total DL (Bytes)") AS total_download,
                SUM("Total UL (Bytes)") AS total_upload,
                SUM("Total DL (Bytes)" + "Total UL (Bytes)") AS total_data_volume
            FROM
                xdr_data
            GROUP BY
                "IMSI", application
            """

        df = self.load_data_from_postgres(query)

        if df is not None and not df.empty:
            # Summarize aggregated data per user
            aggregated_data = df.groupby('user_id').agg(
                num_sessions=('session_count', 'sum'),
                total_duration=('total_duration', 'sum'),
                total_download=('total_download', 'sum'),
                total_upload=('total_upload', 'sum'),
                total_data_volume=('total_data_volume', 'sum')
            ).reset_index()

            # Display aggregated data
            print("User Behavior Analysis:")
            print(aggregated_data)

            # Optional: Return the DataFrame for further processing
            return aggregated_data
        else:
            print("No data available for user behavior analysis.")
            return None
