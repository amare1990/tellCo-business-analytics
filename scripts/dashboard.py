import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
st.title("Interactive Dashboard")

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


# Define plots
def user_overview_plot(data):
    st.header("User Overview Analysis")
    st.bar_chart(data[['User', 'Satisfaction Score']].set_index('User'))




# Render pages
if page == "User Overview Analysis":
    user_overview_plot(data)
