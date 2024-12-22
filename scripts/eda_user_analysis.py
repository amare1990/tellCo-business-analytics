# import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load environment variables from .env file
load_dotenv()

class ExploratoryDataAnalysis:
    def __init__(self):
        # Initialize database connection parameters from environment variables
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")


    # Load data from posgresql database
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

    def treat_missing_values(self):
        """
        Treats missing values in the dataset by replacing them with the mean (or other appropriate methods).

        :return: DataFrame with missing values treated.
        """

        query = "SELECT * FROM xdr_data"
        df = self.load_data_from_postgres(query)

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        # Replace missing values with mean for numerical columns
        # Replace missing values for numerical columns with their mean
        numerical_columns = df.select_dtypes(include=['number']).columns
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

        # Replace missing values for non-numerical columns with a placeholder (e.g., 'Unknown')
        non_numerical_columns = df.select_dtypes(exclude=['number']).columns
        df[non_numerical_columns] = df[non_numerical_columns].fillna('Unknown')

        return df

    def variable_transformations(self):
        """
        Segment users into deciles based on total session duration and compute total data (DL+UL) per decile class.

        :return: DataFrame with transformations and deciles.
        """
        df = self.treat_missing_values()
        # Segment users into top 5 decile classes based on total session duration
        df['total_duration'] = df['Dur. (ms)']
        df['decile'] = pd.qcut(df['total_duration'], 5, labels=False) + 1

        # Compute total data (DL + UL) per decile class
        df['total_data'] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
        decile_data = df.groupby('decile').agg({'total_data': 'sum'}).reset_index()

        return df, decile_data

    def basic_metrics(self):
        """
        Compute basic metrics (mean, median, etc.) for the dataset.

        :return: Basic descriptive statistics.
        """
        df = self.treat_missing_values()

        return df.describe()

    def univariate_analysis(self):
        """
        Conduct non-graphical univariate analysis by computing dispersion parameters for each quantitative variable.

        :return: Dispersion parameters for each variable.
        """
        df, _ = self.variable_transformations()
        # Select only numeric columns for variance calculation
        numeric_df = df.select_dtypes(include='number')

        return numeric_df.var()  # Variance (dispersion)

    def graphical_univariate_analysis(self):
        """
        Conduct a graphical univariate analysis using histograms for each relevant variable.
        """
        df, _ = self.variable_transformations()
        quantitative_cols = ['total_duration', "Total DL (Bytes)", "Total UL (Bytes)", 'total_data']

        for col in quantitative_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, bins=30, color='blue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(f'plots/univariants/hist_univariante_{col}.png', dpi=300, bbox_inches='tight')
            plt.show()



    def bivariate_analysis(self, df):
        """
        Perform bivariate analysis by exploring the relationship between each application's data
        and the total DL+UL data using scatter plots.
        """
        # Ensure any required transformations are applied
        df, _ = self.variable_transformations(df)

        # Define applications with corresponding DL and UL columns
        applications = {
            'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'YouTube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }

        # Compute total data for each application and the overall total
        df['Total Data (DL+UL)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

        for app, columns in applications.items():
            # Calculate total application data (DL + UL)
            df[f'{app} Total'] = df[columns[0]] + df[columns[1]]

            # Plot the relationship between application data and total data (DL+UL)
            plt.figure(figsize=(10, 6))
            plt.scatter(df[f'{app} Total'], df['Total Data (DL+UL)'], alpha=0.7)
            plt.title(f'Relationship between {app} and Total Data (DL+UL)', fontsize=14)
            plt.xlabel(f'{app} Data (DL + UL)', fontsize=12)
            plt.ylabel('Total Data (DL + UL)', fontsize=12)
            plt.grid(True)
            plt.savefig(f'plots/bivariantes/{app}_bivariate_analysis.png', dpi=300, bbox_inches='tight')  # Save the plot
            plt.show()

        print("Bivariate analysis plots saved successfully.")


    def correlation_analysis(self):
        """
        Compute and interpret the correlation matrix for the given application data and plot a pie chart.
        """
        # Ensure any required transformations are applied
        df, _ = self.variable_transformations(df)

        # Define applications with corresponding DL and UL columns
        applications = {
            'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'YouTube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }

        # Calculate total data for each application
        for app, columns in applications.items():
            df[f'{app} Total'] = df[columns[0]] + df[columns[1]]

        # Compute the total DL+UL data (for correlation with all applications)
        df['Total Data (DL+UL)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

        # Define the columns to compute correlation for
        correlation_cols = [f'{app} Total' for app in applications] + ['Total Data (DL+UL)']

        # Compute the correlation matrix
        corr_matrix = df[correlation_cols].corr()

        # Plot the correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')  # Save the plot
        plt.show()

        # Calculate the total data usage per application for the pie chart
        app_totals = {app: df[f'{app} Total'].sum() for app in applications}

        # Plot the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            app_totals.values(),
            labels=app_totals.keys(),
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette('pastel')
        )
        plt.title('Percentage Contribution of Each Application')
        plt.savefig('plots/application_pie_chart.png', dpi=300, bbox_inches='tight')  # Save the plot
        plt.show()


    def dimensionality_reduction(self):
        """
        Perform Principal Component Analysis (PCA) to reduce the dimensionality of the dataset and interpret results.

        :param df: The DataFrame containing the dataset.
        """
        df, _ = self.variable_transformations()
        # Standardizing the data before applying PCA
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        x = df[numeric_features]
        x_scaled = StandardScaler().fit_transform(x)

        # PCA transformation
        pca = PCA(n_components=5)
        principal_components = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['Important1', 'Important2', 'Important3', 'Important4', 'Important5'])

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['Important1'], pca_df['Important2'], pca_df['Important3'], alpha=0.3)
        plt.title('PCA - Dimensionality Reduction')
        plt.xlabel('Important1')
        plt.ylabel('Important2')
        plt.savefig('plots/PCA-dimensionality_reduction.png', dpi=300, bbox_inches='tight')  # Save the plot
        plt.show()

        # PCA interpretation
        explained_variance = pca.explained_variance_ratio_
        print("PCA Interpretation:")
        print(f"1. Important1 explains {explained_variance[0]*100:.2f}% of the variance.")
        print(f"2. Important2 explains {explained_variance[1]*100:.2f}% of the variance.")
