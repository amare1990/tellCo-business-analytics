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


