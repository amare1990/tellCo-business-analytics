import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load data function
def load_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data  # Store the data in session state
        return data
    else:
        st.warning("No file uploaded yet.")
        return pd.DataFrame()  # Return an empty DataFrame

# Check if the data is already loaded in session state
if "data" not in st.session_state:
    data = load_uploaded_file(uploaded_file)
else:
    data = st.session_state.data

# # Display data if loaded
# if not data.empty:
#     st.write("Preview of Uploaded Data:")
#     st.dataframe(data)
# else:
#     st.info("Please upload a CSV file to continue.")


def plot_top_10_handsets(top_10_handsets):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset', data=top_10_handsets, ax=ax, palette='Blues_d')
    ax.set_title('Top 10 Handsets by Usage')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Handset')
    st.pyplot(fig)

def plot_top_3_manufacturers(top_3_manufacturers):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='usage_count', y='manufacturer', data=top_3_manufacturers, ax=ax, palette='viridis')
    ax.set_title('Top 3 Manufacturers by Usage')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Manufacturer')
    st.pyplot(fig)

def plot_top_5_per_manufacturer(top_5_per_manufacturer):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='usage_count', y='handset', data=top_5_per_manufacturer, ax=ax, hue='manufacturer', palette='Set2')
    ax.set_title('Top 5 Handsets per Manufacturer')
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

elif page == "Experience Analysis":
    # Import the UserExperienceAnalyzer class
    from user_experience_analysis import UserExperienceAnalyzer

    # Check if data is loaded
    if data is not None and not data.empty:
        # Initialize the UserExperienceAnalyzer
        experience_analyzer = UserExperienceAnalyzer(data)

        # Aggregating data (required before other analyses)
        aggregated_data = experience_analyzer.aggregate_user_experience_data()

        # Create a sidebar for Experience Analysis options
        st.sidebar.subheader("Experience Analysis Options")
        analysis_option = st.sidebar.radio(
            "Select Analysis Type",
            ["Distribution and Averages per Handset", "K-Means Clustering"]
        )

        if analysis_option == "Distribution and Averages per Handset":
            # Perform analysis and plot visualizations
            pass
            # st.subheader("Distribution and Averages per Handset")
            # results = experience_analyzer.distribution_and_averages_per_handset()

            # # Display plots for throughput and TCP retransmission
            # st.image('plots/user_experience/throughput_per_handset.png', caption="Average Throughput per Handset Type")
            # st.image('plots/user_experience/TCP_per_handset.png', caption="Average TCP Retransmission per Handset Type")

            # # Optionally display the statistics in tabular form
            # st.write("Throughput Distribution:")
            # st.dataframe(results['throughput_distribution'])

            # st.write("Average TCP Retransmission:")
            # st.dataframe(results['tcp_average'])

        elif analysis_option == "K-Means Clustering":
            # K-Means Clustering
            st.subheader("K-Means Clustering of User Experience")
            num_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3)
            cluster_summary, cluster_descriptions = experience_analyzer.kmeans_clustering_user_experience(k=num_clusters)

            # Visualize Clustering Results
            st.write("Cluster Summaries:")
            st.dataframe(cluster_summary)

            # Create scatter plot for clustering
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='RTT', y='Throughput', hue='Cluster', data=data,
                palette='Set2', ax=ax, s=100, alpha=0.8
            )
            ax.set_title("User Clusters (K-Means)")
            ax.set_xlabel("Average RTT")
            ax.set_ylabel("Average Throughput")
            st.pyplot(fig)

            # Display cluster descriptions
            st.write("Cluster Descriptions:")
            for cluster, description in cluster_descriptions.items():
                st.write(description)
    else:
        st.warning("Please upload a CSV file to perform Experience Analysis.")
