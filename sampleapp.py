import streamlit as st
import pandas as pd
#pip install matplotlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
#pip install plotly
import plotly.express as px
from io import StringIO
import os

# Display Company Logo on Every Page
def display_logo():
    st.sidebar.image("acextic-logo.png", width=150)

# Predefined users for Role-Based Access Control (Simulated Login)
users = {
    "admin": {"password": "admin123", "role": "Admin", "name": "Yaswanth", "id": "ADM001"},
    "recruiter": {"password": "recruiter123", "role": "Recruiter", "name": "Kumar", "id": "REC002"}
}

# Inject Custom CSS Styles
st.markdown(
    """
    <style>
    /* Change background color of the main page */
    .main {
        background-color: #f5f5f5;
    }

    /* Customize sidebar background color and text color */
    .css-1d391kg { 
        background-color: #1f77b4 !important;
    }
    .css-10trblm { 
        color: white !important;
    }

    /* Customize headers and titles */
    h1, h2, h3 {
        color: #4b72e0;
    }

    /* Style buttons */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sample Content with New Styles
st.title("Acextic Recruitement")
st.header("Welcome to the Recruitment Insights Dashboard")

# Initialize session state for login and role management
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user"] = None

# Login Function
def login(username, password):
    if username in users and users[username]["password"] == password:
        st.session_state["logged_in"] = True
        st.session_state["user"] = users[username]
        st.success(f"Welcome, {users[username]['name']}!")
    else:
        st.error("Invalid username or password")

# Logout Function
def logout():
    st.session_state["logged_in"] = False
    st.session_state["user"] = None

# Display Login Form if not logged in
if not st.session_state["logged_in"]:
    display_logo()
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        login(username, password)
else:
    # Show User Info and Logout Option
    display_logo()
    user = st.session_state["user"]
    st.sidebar.image("https://via.placeholder.com/100", width=100, caption="Profile Photo")
    st.sidebar.write(f"**Name:** {user['name']}")
    st.sidebar.write(f"**ID:** {user['id']}")
    st.sidebar.write(f"**Role:** {user['role']}")
    if st.sidebar.button("Logout"):
        logout()

# Display Content Based on User Role
if st.session_state["logged_in"]:
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Upload Data", "Search & Filter", "Matches"])

    # Dashboard Tab
    with tab1:
        st.header("Dashboard")
        st.metric(label="Total Consultants", value="150", delta="10%")
        st.metric(label="Successful Placements", value="75", delta="-5%")

        # Plotly Line Chart Example
        data = {'Months': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Placements': [10, 15, 5, 10, 20]}
        df = pd.DataFrame(data)
        fig = px.line(df, x='Months', y='Placements', title='Placements Over Time', markers=True, line_shape='spline')
        fig.update_traces(line=dict(width=3, color='#4b72e0'))
        st.plotly_chart(fig, use_container_width=True)

    # Upload Data Tab
    with tab2:
        st.header("Upload Data")
        with st.form("upload_form"):
            consultant_file = st.file_uploader("Upload Consultant Profiles", type=["csv", "pdf"])
            job_file = st.file_uploader("Upload Job Descriptions", type=["csv", "pdf", "txt", "json"])
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.success("Files uploaded successfully!")
            st.balloons()

    # Search & Filter Tab
    with tab3:
        st.header("Search & Filter")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Consultants")
            skill = st.text_input("Filter by Skill", key="consultant_skill")
            location = st.text_input("Filter by Location", key="consultant_location")

        with col2:
            st.subheader("Jobs")
            job_skill = st.text_input("Filter by Required Skill", key="job_skill")
            job_location = st.text_input("Filter by Location", key="job_location")

    # Matches Tab
    with tab4:
        st.header("Matches")
        if st.button("Match Consultants to Jobs"):
            st.write("### Matching Consultants to Jobs...")
            # Simulated Matching Data
            matches = pd.DataFrame({
                "Consultant": ["Alice", "Bob", "Charlie"],
                "Job": ["Data Scientist", "Backend Developer", "ML Engineer"],
                "Match Percentage": [85, 90, 88]
            })
            st.success("Matches found successfully!")
            st.write(matches)

            if st.button("Send Email Notifications"):
                st.success("Email notifications sent successfully!")

            if st.button("Send SMS Notifications"):
                st.success("SMS notifications sent successfully!")

    # Footer
    st.markdown(
        """
        <footer style="text-align:center; padding:10px;">
        © 2024 - Acextic Recruitment Platform | All rights reserved.
        </footer>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Please log in to access the platform.")
