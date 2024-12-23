import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Data Insights Dashboard", layout="wide")
st.title("tellCo. User Analytics")
st.cache_data.clear()
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")


# st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a Page you want to view",
    [
        "User Overview Analysis",
        "User Engagement Analysis",
        "Experience Analysis",
        "Satisfaction Analysis"
    ]
)


# Load data function
def load_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # st.session_state.data = data  # Store the data in session state
        return data
    else:
        st.warning("No file uploaded yet.")
        return pd.DataFrame()  # Return an empty DataFrame

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

data = load_uploaded_file(uploaded_file)
if not data.empty:
    st.session_state.data = data

# data = load_uploaded_file(uploaded_file)

def plot_top_10_handsets(top_10_handsets):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset', data=top_10_handsets, ax=ax, hue='handset', palette='Blues_d', legend=False)
    ax.set_title('Top 10 Handsets by Usage\n')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Handset')
    st.pyplot(fig)

def plot_top_3_manufacturers(top_3_manufacturers):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='usage_count', y='manufacturer', data=top_3_manufacturers, ax=ax, hue="manufacturer", palette='viridis', legend=False)
    ax.set_title('Top 3 Manufacturers by Usage\n')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Manufacturer')
    st.pyplot(fig)

def plot_top_5_per_manufacturer(top_5_per_manufacturer):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset', data=top_5_per_manufacturer, ax=ax, hue="manufacturer", palette='Set2', legend=True)
    ax.set_title('Top 5 Handsets per Manufacturer\n')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Handset')
    st.pyplot(fig)


# Show analysis based on the selected page
if page == "User Overview Analysis":
    # Importing the UserOverviewAnalyzer inside the conditional block
    from user_overview_analysis import UserOverviewAnalyzer

    # User Overview Analysis Page
    analyzer = UserOverviewAnalyzer()
    analysis_results = analyzer.user_overview_analysis()

    # Ensure the analysis_results are not empty before proceeding
    if analysis_results:
        # Create a sidebar radio button for selecting the type of analysis
        selected_page = st.sidebar.radio(
            "Choose a page for User Overview Analysis",
            ["Top 10 Handsets", "Top 3 Manufacturers", "Top 5 Handsets per Manufacturer"]
        )

        # Display the relevant plot based on the selected page
        if selected_page == "Top 10 Handsets":
            if "top_10_handsets" in analysis_results and not analysis_results["top_10_handsets"].empty:
                plot_top_10_handsets(analysis_results["top_10_handsets"])
            else:
                st.warning("No data available for the Top 10 Handsets analysis.")

        elif selected_page == "Top 3 Manufacturers":
            if "top_3_manufacturers" in analysis_results and not analysis_results["top_3_manufacturers"].empty:
                plot_top_3_manufacturers(analysis_results["top_3_manufacturers"])
            else:
                st.warning("No data available for the Top 3 Manufacturers analysis.")

        elif selected_page == "Top 5 Handsets per Manufacturer":
            if "top_5_per_manufacturer" in analysis_results and not analysis_results["top_5_per_manufacturer"].empty:
                plot_top_5_per_manufacturer(analysis_results["top_5_per_manufacturer"])
            else:
                st.warning("No data available for the Top 5 Handsets per Manufacturer analysis.")
    else:
        st.warning("No data available for visualization.")

elif page == "User Engagement Analysis":
    # Importing the UserEngagementAnalysis inside the conditional block
    from user_engagement_analysis import UserEngagementAnalysis

    # User Engagement Analysis Page
    user_analysis = UserEngagementAnalysis()

    if data is not None and not data.empty:
        # Aggregating the data based on customer ID and metrics
        agg_data, top_customers = user_analysis.aggregate_metrics(data)

        # Aggregating traffic data per application
        app_traffic, top_users_per_app = user_analysis.aggregate_traffic_per_app(data)

        # Plot the top 3 most used applications
        # st.subheader("Top 3 Most Used Applications")
        top_apps = app_traffic.groupby('application')['session_traffic'].sum().nlargest(3)
        fig, ax = plt.subplots(figsize=(6,4))
        top_apps.plot(kind='bar', color=['blue', 'orange', 'green'], ax=ax)
        ax.set_title('Top 3 Most Used Applications')
        ax.set_xlabel('Application')
        ax.set_ylabel('Total Traffic (Bytes)')
        # ax.figure(figsize=(5, 2))
        st.pyplot(fig)

    else:
        st.write("Please upload a CSV file to begin the analysis.")

elif page == "Experience Analysis":
    # Import the UserExperienceAnalyzer class
    from user_experience_analysis import UserExperienceAnalyzer

    # Check if data is loaded
    if data is not None and not data.empty:
        # Initialize the UserExperienceAnalyzer
        experience_analyzer = UserExperienceAnalyzer(data)

        # Aggregating data (required before other analyses)
        aggregated_data = experience_analyzer.aggregate_user_experience_data()

            # K-Means Clustering
        # st.subheader("K-Means Clustering of User Experience")
        num_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3)
        cluster_summary, cluster_descriptions = experience_analyzer.kmeans_clustering_user_experience(4)

        # # Visualize Clustering Results
        # st.write("Cluster Summaries:")
        # st.dataframe(cluster_summary)

        # # Create scatter plot for clustering
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            x='RTT', y='Throughput', hue='Cluster', data=data,
            palette='Set2', ax=ax, s=100, alpha=0.8
        )
        ax.set_title("User Clusters (K-Means)")
        ax.set_xlabel("Average RTT")
        ax.set_ylabel("Average Throughput")
        st.pyplot(fig)

        # # Display cluster descriptions
        # st.write("Cluster Descriptions:")
        # for cluster, description in cluster_descriptions.items():
        #     st.write(description)
    else:
        st.warning("Please upload a CSV file to perform Experience Analysis.")

elif page == "Satisfaction Analysis":
    # Import the UserSatisfactionAnalyzer class
    from user_satisfaction_analysis import UserSatisfactionAnalyzer
    from user_engagement_analysis import UserEngagementAnalysis

    # Check if data is loaded
    if data is not None and not data.empty:
        # Initialize the UserSatisfactionAnalyzer
        satisfaction_analyzer = UserSatisfactionAnalyzer(data)

        # Perform KMeans clustering on engagement and experience scores
        clustered_df = satisfaction_analyzer.kmeans_clustering()

        # Visualize the clusters
        st.subheader("Clustered Satisfaction Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='satisfaction_score',
            y='experience_score',
            hue='cluster',
            data=clustered_df,
            palette='Set2',
            s=100,
            alpha=0.8,
            ax=ax
        )
        ax.set_title("K-Means Clustering on Satisfaction Scores")
        ax.set_xlabel("Satisfaction Score")
        ax.set_ylabel("Experience Score")
        st.pyplot(fig)

        # Display Cluster Summaries
        st.write("Cluster Summaries:")
        st.dataframe(clustered_df)
    else:
        st.warning("Please upload a CSV file to perform Satisfaction Analysis.")
