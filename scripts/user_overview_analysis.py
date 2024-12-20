import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class UserOverviewAnalyzer:
    def __init__(self):
        # Initialize database connection parameters from environment variables
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")

    # def load_data_from_postgres(self, query):
    #     """
    #     Connects to the PostgreSQL database and loads data based on the provided SQL query.

    #     :param query: SQL query to execute.
    #     :return: DataFrame containing the results of the query.
    #     """
    #     try:
    #         # Establish a connection to the database
    #         connection = psycopg2.connect(
    #             host=self.db_host,
    #             port=self.db_port,
    #             database=self.db_name,
    #             user=self.db_user,
    #             password=self.db_password
    #         )

    #         # Load data using pandas
    #         df = pd.read_sql_query(query, connection)

    #         # Close the database connection
    #         connection.close()

    #         return df

    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None


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
                    WHEN "Youtube DL (Bytes)" > 0 OR "Youtube UL (Bytes)" > 0 THEN 'YouTube'
                    WHEN "Netflix DL (Bytes)" > 0 OR "Netflix UL (Bytes)" > 0 THEN 'Netflix'
                    WHEN "Google DL (Bytes)" > 0 OR "Google UL (Bytes)" > 0 THEN 'Google'
                    ELSE 'Other'
                END AS application,
                COUNT("Bearer Id") AS session_count,
                SUM("Dur. (ms)") AS total_duration,
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


class ExploratoryDataAnalysis:
    def __init__(self, db_host, db_port, db_name, db_user, db_password):
        # Initialize database connection parameters
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password

    def load_data_from_postgres(self, query):
        """
        Connects to the PostgreSQL database and loads data based on the provided SQL query.

        :param query: SQL query to execute.
        :return: DataFrame containing the results of the query.
        """
        try:
            # Create an SQLAlchemy engine for PostgreSQL
            engine = create_engine(f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}')

            # Load data using pandas
            df = pd.read_sql_query(query, engine)

            return df

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def treat_missing_values(self, df):
        """
        Treats missing values in the dataset by replacing them with the mean (or other appropriate methods).

        :param df: The DataFrame containing the dataset.
        :return: DataFrame with missing values treated.
        """
        # Replace missing values with mean for numerical columns
        df.fillna(df.mean(), inplace=True)
        return df

    def variable_transformations(self, df):
        """
        Segment users into deciles based on total session duration and compute total data (DL+UL) per decile class.

        :param df: The DataFrame containing the dataset.
        :return: DataFrame with transformations and deciles.
        """
        # Segment users into top 5 decile classes based on total session duration
        df['decile'] = pd.qcut(df['total_duration'], 5, labels=False) + 1

        # Compute total data (DL + UL) per decile class
        df['total_data'] = df['total_dl'] + df['total_ul']
        decile_data = df.groupby('decile').agg({'total_data': 'sum'}).reset_index()

        return df, decile_data

    def basic_metrics(self, df):
        """
        Compute basic metrics (mean, median, etc.) for the dataset.

        :param df: The DataFrame containing the dataset.
        :return: Basic descriptive statistics.
        """
        return df.describe()

    def univariate_analysis(self, df):
        """
        Conduct non-graphical univariate analysis by computing dispersion parameters for each quantitative variable.

        :param df: The DataFrame containing the dataset.
        :return: Dispersion parameters for each variable.
        """
        dispersion = df.var()  # Variance (dispersion)
        return dispersion

    def graphical_univariate_analysis(self, df):
        """
        Conduct a graphical univariate analysis using histograms for each relevant variable.

        :param df: The DataFrame containing the dataset.
        """
        quantitative_cols = ['total_duration', 'total_dl', 'total_ul', 'total_data']

        for col in quantitative_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

    def bivariate_analysis(self, df):
        """
        Perform bivariate analysis by exploring the relationship between each application and the total DL+UL data.

        :param df: The DataFrame containing the dataset.
        """
        applications = ['Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other']
        for app in applications:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[app], y=df['total_data'])
            plt.title(f'Relationship between {app} and Total Data (DL+UL)')
            plt.xlabel(f'{app} Data')
            plt.ylabel('Total Data (DL + UL)')
            plt.show()

    def correlation_analysis(self, df):
        """
        Compute and interpret the correlation matrix for the given application data.

        :param df: The DataFrame containing the dataset.
        """
        correlation_cols = ['Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other']
        corr_matrix = df[correlation_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

    def dimensionality_reduction(self, df):
        """
        Perform Principal Component Analysis (PCA) to reduce the dimensionality of the dataset and interpret results.

        :param df: The DataFrame containing the dataset.
        """
        # Standardizing the data before applying PCA
        from sklearn.preprocessing import StandardScaler
        features = ['total_duration', 'total_dl', 'total_ul', 'total_data']
        x = df[features]
        x_scaled = StandardScaler().fit_transform(x)

        # PCA transformation
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['PC1'], pca_df['PC2'])
        plt.title('PCA - Dimensionality Reduction')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        # PCA interpretation
        explained_variance = pca.explained_variance_ratio_
        print("PCA Interpretation:")
        print(f"1. PC1 explains {explained_variance[0]*100:.2f}% of the variance.")
        print(f"2. PC2 explains {explained_variance[1]*100:.2f}% of the variance.")
        print("3. The data is reduced to two principal components for easy visualization.")
        print("4. PCA highlights the most influential variables in the dataset.")
