import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import mysql.connector

from scripts.user_engagement_analysis import UserEngagementAnalysis
from scripts.user_experience_analysis import UserExperienceAnalyzer

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
