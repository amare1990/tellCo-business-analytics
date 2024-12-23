import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the UserOverviewAnalyzer and UserEngagementAnalysis classes
from user_overview_analysis import UserOverviewAnalyzer
from user_engagement_analysis import UserEngagementAnalysis

# Configure the Streamlit page
st.set_page_config(page_title="Data Insights Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a Page",
    [
        "User Overview Analysis",
        "User Engagement Analysis",
        "Experience Analysis",
        "Satisfaction Analysis"
    ]
)


# Streamlit app title
st.title("tellCo. User Analytics")

# File uploader widget
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Load data
def load_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.warning("No file uploaded yet.")
        return pd.DataFrame()  # Return an empty DataFrame

data = load_uploaded_file(uploaded_file)

# Display data if loaded
if not data.empty:
    st.write("Preview of Uploaded Data:")
    st.dataframe(data)
else:
    st.info("Please upload a CSV file to continue.")

# Show analysis based on the selected page
if page == "User Overview Analysis":
    # User Overview Analysis Page
    analyzer = UserOverviewAnalyzer()
    analysis_results = analyzer.user_overview_analysis()

    if analysis_results:
        st.sidebar.radio(
            "Choose a page for User Overview Analysis",
            ["Top 10 Handsets", "Top 3 Manufacturers", "Top 5 Handsets per Manufacturer"]
        )

        selected_page = st.sidebar.radio(
            "Choose a page",
            ["Top 10 Handsets", "Top 3 Manufacturers", "Top 5 Handsets per Manufacturer"]
        )

        if selected_page == "Top 10 Handsets":
            plot_top_10_handsets(analysis_results["top_10_handsets"])
        elif selected_page == "Top 3 Manufacturers":
            plot_top_3_manufacturers(analysis_results["top_3_manufacturers"])
        elif selected_page == "Top 5 Handsets per Manufacturer":
            plot_top_5_per_manufacturer(analysis_results["top_5_per_manufacturer"])
    else:
        st.warning("No data available for visualization.")

elif page == "User Engagement Analysis":
    # User Engagement Analysis Page
    user_analysis = UserEngagementAnalysis()

    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        data = pd.read_csv(uploaded_file)

        # Aggregating the data based on customer ID and metrics
        agg_data, top_customers = user_analysis.aggregate_metrics(data)

        # Aggregating traffic data per application
        app_traffic, top_users_per_app = user_analysis.aggregate_traffic_per_app(data)

        # Plot the top 3 most used applications
        st.subheader("Top 3 Most Used Applications")
        top_apps = app_traffic.groupby('application')['session_traffic'].sum().nlargest(3)
        fig, ax = plt.subplots()
        top_apps.plot(kind='bar', color=['blue', 'orange', 'green'], ax=ax)
        ax.set_title('Top 3 Most Used Applications')
        ax.set_xlabel('Application')
        ax.set_ylabel('Total Traffic (Bytes)')
        st.pyplot(fig)

        # Optionally, display the top 3 users per app as a table
        st.subheader("Top 10 Users per Application")
        st.write(top_users_per_app)
    else:
        st.write("Please upload a CSV file to begin the analysis.")
