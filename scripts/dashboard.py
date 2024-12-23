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

import streamlit as st
import pandas as pd

# File uploader function
@st.cache_data
def load_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Assuming the uploaded file is a CSV
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.warning("No file uploaded yet.")
        return pd.DataFrame()  # Return an empty DataFrame

# Streamlit app
st.title("tellCo. User Analytics")

# File uploader widget
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Load data
data = load_uploaded_file(uploaded_file)

# Display data
if not data.empty:
    st.write("Preview of Uploaded Data:")
    st.dataframe(data)
    # Add your visualization logic here
else:
    st.info("Please upload a CSV file to continue.")


# Define a function for visualizing top 10 handsets
def plot_top_10_handsets(data):
    st.subheader("Top 10 Handsets")
    if data is not None and not data.empty:
        fig, ax = plt.subplots()
        sns.barplot(data=data, x="usage_count", y="handset", ax=ax, palette="Blues_r")
        ax.set_title("Top 10 Handsets by Usage")
        ax.set_xlabel("Usage Count")
        ax.set_ylabel("Handset Type")
        st.pyplot(fig)
    else:
        st.warning("No data available for Top 10 Handsets.")

# Define a function for visualizing top 3 manufacturers
def plot_top_3_manufacturers(data):
    st.subheader("Top 3 Handset Manufacturers")
    if data is not None and not data.empty:
        fig, ax = plt.subplots()
        sns.barplot(data=data, x="usage_count", y="manufacturer", ax=ax, palette="Greens_r")
        ax.set_title("Top 3 Manufacturers by Usage")
        ax.set_xlabel("Usage Count")
        ax.set_ylabel("Manufacturer")
        st.pyplot(fig)
    else:
        st.warning("No data available for Top 3 Manufacturers.")

# Define a function for visualizing top 5 handsets per manufacturer
def plot_top_5_per_manufacturer(data):
    st.subheader("Top 5 Handsets per Manufacturer")
    if data is not None and not data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=data,
            x="usage_count",
            y="handset",
            hue="manufacturer",
            dodge=False,
            ax=ax,
            palette="coolwarm"
        )
        ax.set_title("Top 5 Handsets by Top 3 Manufacturers")
        ax.set_xlabel("Usage Count")
        ax.set_ylabel("Handset")
        st.pyplot(fig)
    else:
        st.warning("No data available for Top 5 Handsets per Manufacturer.")

# Import your UserOverviewAnalyzer class
from user_overview_analysis import UserOverviewAnalyzer

# Initialize the analyzer
analyzer = UserOverviewAnalyzer()
# Run the analysis and get the results
analysis_results = analyzer.user_overview_analysis()

# Visualize the results
if analysis_results:
    st.sidebar.title("Navigation")
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


# Footer
st.sidebar.write("Dashboard by Your Amare Mekonnen")

# # Render pages
# if page == "User Overview Analysis":
#     user_overview_plot(data)
