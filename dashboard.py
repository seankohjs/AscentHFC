import streamlit as st
import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from load_and_process import process_data, get_all_feedback_data, summarize_feedback

# Page configuration
st.set_page_config(
    page_title="Feedback Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("Feedback Analysis Dashboard")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.write("Use the options below to explore the dashboard.")
selected_section = st.sidebar.radio("Go to", ["Overview Report", "Visual Charts", "Settings"])

# Date Range Selector
st.sidebar.markdown("---")
st.sidebar.header("Data Filters")
start_date = st.sidebar.date_input("Start Date", datetime.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

if start_date > end_date:
  st.sidebar.error("Start date must be before end date!")

# Main content
if selected_section == "Overview Report":
    # Load and Process Data
    df = get_all_feedback_data(start_date, end_date)
    if not df.empty: # Check if there is data
      overall_sentiment, total_feedback, positive_feedback, negative_feedback, category_counts, segments, df_monthly = process_data(df)
      ai_summary = summarize_feedback(df)
    else:
      overall_sentiment = 0
      total_feedback = 0
      positive_feedback = 0
      negative_feedback = 0
      ai_summary = "No Feedback Found"

    # Massive header on the top left
    st.header("Feedback Overview")
    st.write("This section provides a high-level summary of all collected feedback.")
    st.markdown("---")

    # Section for the first row
    st.subheader(f"Average Sentiment: {overall_sentiment:.1f} | Total Feedback Count: {total_feedback}")
    st.write("This provides the average sentiment as well as the total number of feedback collected")
    st.markdown("---")

    # Section for the second row
    st.subheader("AI Overall Summary of all feedback")
    st.write(ai_summary)
    st.markdown("---")

    st.subheader("Feedback Counts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("All Feedback", total_feedback, "+5%")
    with col2:
        st.metric("Positive Feedback", positive_feedback, "-3%")
    with col3:
        st.metric("Negative Feedback", negative_feedback, "2%")
    st.markdown("---")

elif selected_section == "Visual Charts":
    st.header("Feedback Visualizations")
    
    # Load and Process Data
    df = get_all_feedback_data(start_date, end_date)
    if not df.empty:
        overall_sentiment, total_feedback, positive_feedback, negative_feedback, category_counts, segments, df_monthly = process_data(df)
    else:
      overall_sentiment = 0
      total_feedback = 0
      category_counts = {}
      segments = {}
      df_monthly = pd.DataFrame()
    # Top metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Sentiment Score", f"{overall_sentiment:.2f}", "+8.2%")
    with col2:
        st.metric("Total Feedback Count", f"{total_feedback}", "+2.1%")


    st.markdown("---")

    # Two charts in one row
    col_left, col_right = st.columns(2)

    with col_left:
        # Feedback counts by Category
        st.subheader("Feedback Counts by Category")
        fig_categories = go.Figure(go.Bar(
            x=list(category_counts.values()),
            y=list(category_counts.keys()),
            orientation='h',
            marker_color='#2C7FB8'
        ))
        fig_categories.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Feedback Count",
            height=300
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    with col_right:
        # Gross Sales by Segment
        st.subheader("Feedback Ratio")
        fig_segments = go.Figure(data=[go.Pie(
            labels=list(segments.keys()),
            values=list(segments.values()),
            hole=.6
        )])
        fig_segments.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            height=300
        )
        st.plotly_chart(fig_segments, use_container_width=True)

    st.markdown("---")

    # Time series chart
    st.subheader("Total Number of Feedbacks")
    
    if not df_monthly.empty:
      fig = go.Figure()
      fig.add_trace(go.Scatter(
          x=df_monthly['Date'],
          y=df_monthly['Feedback Count'],
          name="Total Feedback",
          line=dict(color="#00CC96", width=2),
          fill='tozeroy'
      ))

      fig.update_layout(
          height=400,
          margin=dict(l=0, r=0, t=20, b=0),
          legend=dict(
              orientation="h",
              yanchor="bottom",
              y=1.02,
              xanchor="right",
              x=1
          )
      )
      st.plotly_chart(fig, use_container_width=True)


elif selected_section == "Settings":
    st.subheader("Settings Section")
    st.write("Modify application settings here.")
    theme = st.selectbox("Choose theme", ["Light", "Dark"])
    notifications = st.checkbox("Enable notifications", value=True)
    st.button("Save Settings")

# Footer
st.write("Developed with ❤️ using Streamlit")