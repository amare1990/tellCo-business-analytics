import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.user_engagement_analysis import UserEngagementAnalysis
from scripts.user_experience_analysis import UserExperienceAnalyzer

class UserSatisfactionAnalyzer:
    def __init__(self, data):
        """
        Initialize the UserAnalytics class with the dataset.

        Parameters:
        data (pd.DataFrame): The dataset containing network parameters and user information.
        """
        self.data = data
        self.user_engagement = UserEngagementAnalysis()
        self.user_experience = UserExperienceAnalyzer(data)

    def aggregate_and_normalize(self):
        """Aggregate user engagement data and normalize for clustering."""
        agg_data, _ = self.user_engagement.aggregate_metrics(self.data)
        # Normalize data for clustering
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(agg_data[['session_frequency', 'session_duration', 'session_traffic']])
        return agg_data, normalized_data
