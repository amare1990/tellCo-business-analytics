import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from scripts.user_engagement_analysis import UserEngagementAnalysis
from scripts.user_experience_analysis import UserExperienceAnalyzer

# Load environment variables from .env file
load_dotenv()

# Define the base class for our SQLAlchemy model
Base = declarative_base()


class UserSatisfactionAnalyzer:
    def __init__(self, data):
        """
        Initialize the UserSatisfactionAnalyzer with the dataset.

        Parameters:
        data (pd.DataFrame): The dataset containing network parameters and user information.
        """
        self.data = data
        self.engagement_scores = None
        self.experience_scores = None

        # Initialize database connection parameters from environment variables
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")

        # Database connection URL
        self.db_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"




    def normalize_and_cluster(self, data, k=3, n_init=10):
      """Normalize data and perform k-means clustering."""
      # Select only numeric columns
      numeric_data = data.select_dtypes(include=['float64', 'int64'])

      # Fill NaN values with the mean of the respective columns
      data.loc[:, numeric_data.columns] = numeric_data.fillna(numeric_data.mean())

      # Normalize the data using MinMaxScaler
      scaler = MinMaxScaler()
      normalized_data = scaler.fit_transform(data[numeric_data.columns])

      # Perform k-means clustering
      kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
      clusters = kmeans.fit_predict(normalized_data)

      return normalized_data, kmeans, clusters


    def assign_engagement_score(self):
        """Assign engagement scores based on Euclidean distance from the least engaged cluster."""
        # Aggregate metrics
        user_engagement = UserEngagementAnalysis()
        agg_data, _ = user_engagement.aggregate_metrics(self.data)

        # Normalize and cluster
        normalized_data, kmeans, clusters = self.normalize_and_cluster(
            agg_data[['session_frequency', 'session_duration', 'session_traffic']]
        )

        # Assign cluster labels
        agg_data['cluster'] = clusters

        # Identify the least engaged cluster (based on sum of engagement metrics)
        cluster_centers = kmeans.cluster_centers_
        least_engaged_cluster_idx = np.argmin(cluster_centers.sum(axis=1))

        # Calculate engagement scores
        least_engaged_center = cluster_centers[least_engaged_cluster_idx]
        agg_data['engagement_score'] = euclidean_distances(
            normalized_data, least_engaged_center.reshape(1, -1)
        ).flatten()

        self.engagement_scores = agg_data[['MSISDN/Number', 'engagement_score']]
        return self.engagement_scores



    def assign_experience_score(self):
        """Assign experience scores based on Euclidean distance from the worst experience cluster."""
        user_experience = UserExperienceAnalyzer(self.data)

        # Aggregate user experience data
        experience_data = user_experience.aggregate_user_experience_data()

        # Normalize and cluster
        normalized_data, kmeans, clusters = self.normalize_and_cluster(experience_data)

        # Assign cluster labels
        experience_data['cluster'] = clusters

        # Identify the worst experience cluster (based on specific domain logic, e.g., lowest performance metrics)
        cluster_centers = kmeans.cluster_centers_
        worst_experience_cluster_idx = np.argmin(cluster_centers.mean(axis=1))
        worst_experience_center = cluster_centers[worst_experience_cluster_idx]

        # Calculate experience scores
        experience_data['experience_score'] = euclidean_distances(
            normalized_data, worst_experience_center.reshape(1, -1)
        ).flatten()

        self.experience_scores = experience_data[['MSISDN/Number', 'experience_score']]
        return self.experience_scores


    def analyze_user_satisfaction(self):
        """Combine engagement and experience scores into a single DataFrame."""
        if self.engagement_scores is None or self.experience_scores is None:
            raise ValueError("Engagement and experience scores must be computed first.")

        satisfaction_df = pd.merge(
            self.engagement_scores, self.experience_scores, on='MSISDN/Number', how='inner'
        )
        merged_data = satisfaction_df
        satisfaction_df['satisfaction_score'] = (
            satisfaction_df['engagement_score'] + satisfaction_df['experience_score']) / 2

        # Get top 10 satisfied customers
        top10_customers = satisfaction_df.nlargest(10, 'satisfaction_score')

        return merged_data, satisfaction_df, top10_customers


    def train_regression_model(self):
        """Trains a regression model to predict the satisfaction score of a customer
           based on engagement and experience score
        """
        X, satisfaction_df, _= self.analyze_user_satisfaction()
        y = satisfaction_df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(f"Regression Model Coefficients: {model.coef_}")
        print(f"Regression Model Intercept: {model.intercept_}")
        return model


    def kmeans_clustering(self):
      """Performs k-means clustering on the engagement and experience scores"""
      # Get the merged data and satisfaction scores from analyze_user_satisfaction
      merged_data, satisfaction_df, _ = self.analyze_user_satisfaction()

      # Perform KMeans clustering on the engagement and experience scores
      kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
      merged_data['cluster'] = kmeans.fit_predict(merged_data[['engagement_score', 'experience_score']])

      # Clustered data: Group by 'cluster' and calculate mean satisfaction and experience scores
      clustered_df = merged_data.groupby('cluster').agg({
          'satisfaction_score': 'mean',
          'experience_score': 'mean'
      }).reset_index()

      return clustered_df



    def export_to_postgresql(self):
        """Exports the final table to a PostgreSQL database using SQLAlchemy"""
        # Create SQLAlchemy engine
        engine = create_engine(self.db_url)

        # Create all tables in the database
        Base.metadata.create_all(engine)

        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()

        _, satisfaction_df, _= self.analyze_user_satisfaction()

        # Insert data into the database
        for index, row in self.df.iterrows():
            user_satisfaction = UserSatisfactionAnalyzer(
                user_id=row['Bearer Id'],
                engagement_score=row['engagement_score'],
                experience_score=row['experience_score'],
                satisfaction_score=row['satisfaction_score'],
                cluster=row['cluster']
            )
            session.add(user_satisfaction)

        # Commit the transaction
        session.commit()

        # Close the session
        session.close()

        print("Data exported successfully to PostgreSQL database.")




