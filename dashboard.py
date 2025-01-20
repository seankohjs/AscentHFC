import streamlit as st
import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

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

# might be useful later on for filtering for report
# st.sidebar.markdown("---")
# st.sidebar.header("Data Filters")
# start_date = st.sidebar.date_input("Start Date", datetime.date(2025, 1, 1))
# end_date = st.sidebar.date_input("End Date", datetime.date.today())

# if start_date > end_date:
#     st.sidebar.error("Start date must be before end date!")

# Main content
if selected_section == "Overview Report":
    # Sample Data for Overview Section
    average_rating = 3.8
    ai_summary = "AI analysis indicates that while users appreciate the intent of the government schemes, there are concerns about the clarity of information and the level of support provided. Many felt they were complicated and not easy to understand. Some users appreciated the financial help."
    total_feedback_count = 456
    positive_feedback_count = 333
    negative_feedback_count = 123
  
    # Massive header on the top left
    st.header("Feedback Overview")
    st.write("This section provides a high-level summary of all collected feedback.")
    st.markdown("---")

    # Section for the first row
    st.subheader(f"Average Rating: {average_rating:.1f} | Total Feedback Count: {total_feedback_count}")
    st.write("This provides the average ratings as well as the total number of feedback collected")
    st.markdown("---")

    # Section for the second row
    st.subheader("AI Overall Summary of all feedback")
    st.write(ai_summary)
    st.markdown("---")

    st.subheader("Feedback Counts")
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("All Feedback", total_feedback_count, "+5%")
    with col2:
        st.metric("Positive Feedback", positive_feedback_count, "-3%")
    with col3:
        st.metric("Negative Feedback", negative_feedback_count, "2%")
    st.markdown("---")

elif selected_section == "Visual Charts":
    st.header("Feedback Visualizations")
    
    # Create sample data, will be changed accordingly to fit the correct data inputs
    def generate_sample_data():
      #Sample Metrics
        metrics = {
          'Overall Sentiment Score': 0.3,
          'Total Feedback': 456,
          }
        
        #Sample Category Data
        category_counts = {
            "Scheme Specific Feedback" : 200,
            "General Feedback" : 150,
            "Chatbot Feedback" : 100
        }
        
        #Sample Sentiment Data
        segments = {
            'Positive': 56.40,
            'Neutral': 45.94,
            'Negative': 21.07
        }
        
        # Sample Time Series Data
        dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='M')
        feedback_counts = np.random.normal(20, 5, len(dates)) + np.sin(np.arange(len(dates)) * 0.3) * 3
        
        df = pd.DataFrame({
          'Date': dates,
          'Feedback Count': feedback_counts,
        })
        
        return metrics, category_counts, segments, df
      

    # Get sample data
    metrics, category_counts, segments, df = generate_sample_data()

    # Top metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Sentiment Score", f"{metrics['Overall Sentiment Score']:.2f}", "+8.2%")
    with col2:
        st.metric("Total Feedback Count", f"{metrics['Total Feedback']}", "+2.1%")


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
    st.subheader("Feedback Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Feedback Count'],
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