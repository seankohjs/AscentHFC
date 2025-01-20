import streamlit as st
import datetime
import plotly.graph_objects as go
import pandas as pd
import os
from functions import process_data, get_all_feedback_data, summarize_feedback
from datetime import date, timedelta

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
selected_section = st.sidebar.radio("Go to", ["Preprocess Data", "Overview Report", "Visual Charts", "Settings"])

# --- Data Loading Logic ---
def load_preprocessed_data(filename):
    try:
        # Construct the full file path
        file_path = os.path.join("data/preprocessed", filename)
        if filename.endswith(".csv"):
           df = pd.read_csv(file_path)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file type")
        return df
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file {filename}: {e}")
        return pd.DataFrame()

def save_preprocessed_data(df, start_date, end_date):
    # Format the date range to create the filename
    filename = f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
    os.makedirs("data/preprocessed", exist_ok=True)
    file_path_parquet = os.path.join("data/preprocessed", f"{filename}.parquet")
    file_path_csv = os.path.join("data/preprocessed", f"{filename}.csv")
    
    try:
        # Generate the AI summary before saving
        ai_summary = summarize_feedback(df)
        df['ai_summary'] = ai_summary # Add the ai_summary as a column
        df.to_parquet(file_path_parquet, index = False)
        df.to_csv(file_path_csv, index=False)
        st.success(f"Data saved to {filename}.parquet and {filename}.csv")
    except Exception as e:
        st.error(f"Error saving data: {e}")

def delete_preprocessed_data(filename):
    file_path = os.path.join("data/preprocessed", filename)
    try:
      os.remove(file_path)
      if filename.endswith(".parquet"):
         os.remove(file_path.replace(".parquet", ".csv"))
      elif filename.endswith(".csv"):
        os.remove(file_path.replace(".csv", ".parquet"))
      st.success(f"Successfully deleted: {filename}")
    except FileNotFoundError:
        st.error(f"Error: File not found {filename}")
    except Exception as e:
        st.error(f"Error deleting file {filename}: {e}")

# --- Preprocess Data Page ---
if selected_section == "Preprocess Data":
    st.header("Preprocess Data")
    st.write("Select a date range to preprocess feedback data:")

    # Date Range Selector
    start_date = st.date_input("Start Date", date.today() - timedelta(days=7))
    end_date = st.date_input("End Date", date.today())

    if start_date > end_date:
        st.error("Start date must be before end date!")
    else:
        # File Exists Check
        filename = f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
        file_path_parquet = os.path.join("data/preprocessed", f"{filename}.parquet")
        file_path_csv = os.path.join("data/preprocessed", f"{filename}.csv")
        if os.path.exists(file_path_parquet) and os.path.exists(file_path_csv):
          st.warning(f"A preprocessed data file already exists for {filename}. You can delete it below, or select a new date range.")
        else:
          if st.button("Process Data"):
              with st.spinner("Processing data..."):
                  df = get_all_feedback_data(start_date, end_date)
                  if not df.empty:
                    save_preprocessed_data(df, start_date, end_date)
                  else:
                      st.warning("No feedback found for the selected date range.")

    # List existing files
    st.subheader("Existing Preprocessed Data")
    preprocessed_files = [f for f in os.listdir("data/preprocessed") if f.endswith((".parquet", ".csv"))]
    if preprocessed_files:
        selected_files_to_delete = st.multiselect("Select files to delete", preprocessed_files)
        if st.button("Delete Selected Files"):
          for file_to_delete in selected_files_to_delete:
            delete_preprocessed_data(file_to_delete)
    else:
        st.write("No preprocessed data found.")
    
# --- Overview Report Page ---
elif selected_section == "Overview Report":
    st.header("Feedback Overview")
    st.write("This section provides a high-level summary of all collected feedback.")
    st.markdown("---")

    # Load Preprocessed Data
    preprocessed_files = [f for f in os.listdir("data/preprocessed") if f.endswith((".parquet", ".csv"))]
    if preprocessed_files:
      selected_file = st.selectbox("Select preprocessed data:", preprocessed_files)
      if selected_file:
        df = load_preprocessed_data(selected_file)
    else:
      df = pd.DataFrame()
      st.warning("No preprocessed data available, please process the data in the Preprocess Data tab")

    if not df.empty:
        overall_sentiment, total_feedback, positive_feedback, negative_feedback, category_counts, segments, df_monthly = process_data(df)
        ai_summary = df['ai_summary'].iloc[0]
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
    else:
         st.warning("Please select a valid preprocessed data file.")

# --- Visual Charts Page ---
elif selected_section == "Visual Charts":
    st.header("Feedback Visualizations")

    # Load Preprocessed Data
    preprocessed_files = [f for f in os.listdir("data/preprocessed") if f.endswith((".parquet", ".csv"))]
    if preprocessed_files:
      selected_file = st.selectbox("Select preprocessed data:", preprocessed_files)
      if selected_file:
        df = load_preprocessed_data(selected_file)
    else:
      df = pd.DataFrame()
      st.warning("No preprocessed data available, please process the data in the Preprocess Data tab")

    if not df.empty:
        overall_sentiment, total_feedback, positive_feedback, negative_feedback, category_counts, segments, df_monthly = process_data(df)
    
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
    else:
        st.warning("Please select a valid preprocessed data file.")

# --- Settings Page ---
elif selected_section == "Settings":
    st.subheader("Settings Section")
    st.write("Modify application settings here.")
    theme = st.selectbox("Choose theme", ["Light", "Dark"])
    notifications = st.checkbox("Enable notifications", value=True)
    st.button("Save Settings")

# Footer
st.write("Developed with ❤️ using Streamlit")