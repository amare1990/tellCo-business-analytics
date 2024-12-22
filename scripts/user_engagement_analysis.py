import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class UserEngagementAnalysis:
    def __init__(self):
        # Initialize the class
        pass

    def aggregate_metrics(self, data):
      """Aggregate metrics per customer id."""
      # Rename columns to clarify
      data = data.rename(columns={
          'Dur. (ms)': 'session_duration',
          'Total UL (Bytes)': 'upload_traffic',
          'Total DL (Bytes)': 'download_traffic'
      })

      # Derive session_frequency and session_traffic
      data['session_frequency'] = data.groupby('MSISDN/Number')['Bearer Id'].transform('count')
      data['session_traffic'] = data['upload_traffic'] + data['download_traffic']

      # Aggregate data per customer
      agg_data = data.groupby("MSISDN/Number").agg({
          'session_frequency': 'sum',
          'session_duration': 'sum',
          'session_traffic': 'sum'
      }).reset_index()

      # Get top 10 customers for each metric
      top_customers = {
          'session_frequency': agg_data.nlargest(10, 'session_frequency'),
          'session_duration': agg_data.nlargest(10, 'session_duration'),
          'session_traffic': agg_data.nlargest(10, 'session_traffic')
      }
      return agg_data, top_customers


    def normalize_and_cluster(self, agg_data, k=3, n_init=10):
        """Normalize metrics and perform k-means clustering."""
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(agg_data[['session_frequency', 'session_duration', 'session_traffic']])
        kmeans = KMeans(n_clusters=k, random_state=4, n_init=n_init)
        agg_data['cluster'] = kmeans.fit_predict(normalized_data)
        return agg_data, kmeans

    def compute_cluster_statistics(self, agg_data):
        """Compute statistics for each cluster."""
        stats = agg_data.groupby('cluster').agg({
            'session_frequency': ['min', 'max', 'mean', 'sum'],
            'session_duration': ['min', 'max', 'mean', 'sum'],
            'session_traffic': ['min', 'max', 'mean', 'sum']
        })
        return stats

    def aggregate_traffic_per_app(self, data):
        """Aggregate user total traffic per application."""
        applications = {
            'Social Media': ["Social Media DL (Bytes)", 'Social Media UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'YouTube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }

        # Initialize an empty DataFrame for aggregated traffic
        app_traffic = []

        # Loop through each application and aggregate traffic
        for app_name, columns in applications.items():
            data[f"{app_name}_traffic"] = data[columns[0]] + data[columns[1]]
            app_df = (
                data.groupby("MSISDN/Number")[[f"{app_name}_traffic"]]
                .sum()
                .reset_index()
            )
            app_df["application"] = app_name
            app_df.rename(columns={f"{app_name}_traffic": "session_traffic"}, inplace=True)
            app_traffic.append(app_df)

        # Concatenate traffic data for all applications
        app_traffic = pd.concat(app_traffic, ignore_index=True)

        # Get top users per application
        top_users_per_app = app_traffic.sort_values("session_traffic", ascending=False).groupby("application").head(10)

        return app_traffic, top_users_per_app


    def plot_top_apps(self, app_traffic):
        """Plot the top 3 most used applications."""
        top_apps = app_traffic.groupby('application')['session_traffic'].sum().nlargest(3)
        top_apps.plot(kind='bar', color=['blue', 'orange', 'green'])
        plt.title('Top 3 Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        plt.savefig("plots/top_apps.png", dpi=300, bbox_inches='tight')
        plt.show()

    def determine_optimal_k(self, agg_data):
        """Use the elbow method to determine the optimal number of clusters."""
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(agg_data[['session_frequency', 'session_duration', 'session_traffic']])
        distortions = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(normalized_data)
            distortions.append((kmeans.inertia_))

        plt.figure(figsize=(8, 5))
        plt.plot(K, distortions, 'bo-', markersize=8)
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        plt.savefig("plots/optimal_k.png", dpi=300, bbox_inches='tight')
        plt.show()

        return distortions
