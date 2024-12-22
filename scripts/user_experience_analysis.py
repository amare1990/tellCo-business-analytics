import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns



class UserExperienceAnalyzer:
    def __init__(self, data):
        """
        Initialize the ExperienceAnalytics class with the dataset.

        Parameters:
        data (pd.DataFrame): The dataset containing network parameters and user information.
        """
        self.data = data

    def aggregate_user_experience_data(self):
        """
        Aggregate per customer:
        - Average TCP retransmission
        - Average RTT (downlink and uplink combined)
        - Average throughput (downlink and uplink combined)
        - Mode of Handset Type

        Handles missing values and outliers by replacing them with mean/mode.

        Returns:
        pd.DataFrame: Aggregated data per customer.
        """
        # Calculate derived metrics
        self.data['TCP Retransmission'] = (
            self.data['TCP DL Retrans. Vol (Bytes)'] + self.data['TCP UL Retrans. Vol (Bytes)']
        )
        self.data['RTT'] = (
            self.data['Avg RTT DL (ms)'] + self.data['Avg RTT UL (ms)']
        ) / 2
        self.data['Throughput'] = (
            self.data['Avg Bearer TP DL (kbps)'] + self.data['Avg Bearer TP UL (kbps)']
        )

        # Handle missing values
        self.data.fillna({
            'TCP Retransmission': self.data['TCP Retransmission'].mean(),
            'RTT': self.data['RTT'].mean(),
            'Throughput': self.data['Throughput'].mean(),
            'Handset Type': self.data['Handset Type'].mode()[0]
        }, inplace=True)

        # List of columns to handle outliers
        outlier_columns = ['TCP Retransmission', 'RTT', 'Throughput']

        for col in outlier_columns:
          # Calculate IQR
          Q1 = self.data[col].quantile(0.25)
          Q3 = self.data[col].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR

          # Calculate mean or mode for replacement
          if self.data[col].dtype in ['float64', 'int64']:  # Continuous data
              replacement_value = self.data[col].mean()
          else:  # Categorical data
              replacement_value = self.data[col].mode()[0]

          # Replace outliers with mean or mode
          self.data[col] = self.data[col].apply(
              lambda x: replacement_value if (x < lower_bound or x > upper_bound) else x
          )


        # Aggregation
        aggregated = self.data.groupby('MSISDN/Number').agg({
            'TCP Retransmission': 'mean',
            'RTT': 'mean',
            'Throughput': 'mean',
            'Handset Type': lambda x: x.mode()[0]
        }).reset_index()

        return aggregated


    def compute_top_bottom_frequent(self):
        """
        Compute and list 10 of the top, bottom, and most frequent:
        - TCP values
        - RTT values
        - Throughput values

        Returns:
        dict: A dictionary with top, bottom, and frequent values for each metric.
        """
        results = {}
        for column in ['TCP Retransmission', 'RTT', 'Throughput']:
            sorted_values = self.data[column].sort_values()
            results[column] = {
                'Top_10 values': sorted_values.tail(10).tolist(),
                'Bottom_10 values': sorted_values.head(10).tolist(),
                'Most_frequent values': self.data[column].value_counts().head(10).index.tolist()
            }
        return results

    def distribution_and_averages_per_handset(self):
        """
        Compute and report:
        - Distribution of average throughput per handset type.
        - Average TCP retransmission per handset type.

        Returns:
        dict: Distribution and average statistics for throughput and TCP retransmission.
        """
        throughput_distribution = self.data.groupby('Handset Type')['Throughput'].mean()
        tcp_average = self.data.groupby('Handset Type')['TCP Retransmission'].mean()

        # Plotting
        plt.figure(figsize=(12, 6))
        sns.barplot(x=throughput_distribution.index, y=throughput_distribution.values)
        plt.title('Average Throughput per Handset Type')
        plt.xlabel('Handset Type')
        plt.ylabel('Average Throughput (kbps)')
        plt.xticks(rotation=45)
        plt.savefig('plots/user_experience/throughput_per_handset.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=tcp_average.index, y=tcp_average.values)
        plt.title('Average TCP Retransmission per Handset Type')
        plt.xlabel('Handset Type')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.xticks(rotation=45)
        plt.savefig('plots/user_experience/TCP_per_handset.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {'throughput_distribution': throughput_distribution, 'tcp_average': tcp_average}


    def kmeans_clustering_user_experience(self, k=3):
        """
        Perform k-means clustering to segment users into groups of experiences.

        Parameters:
        k (int): Number of clusters. Default is 3.

        Returns:
        pd.DataFrame: Data with cluster labels.
        dict: Cluster descriptions.
        """
        # Prepare data
        features = self.data[['TCP Retransmission', 'RTT', 'Throughput']]
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.data['Cluster'] = kmeans.fit_predict(normalized_data)

        # Cluster descriptions
        cluster_summary = self.data.groupby('Cluster')[['TCP Retransmission', 'RTT', 'Throughput']].mean()
        cluster_descriptions = {
            i: f"Cluster {i}: {row.to_dict()}" for i, row in cluster_summary.iterrows()
        }

        return cluster_summary, cluster_descriptions
