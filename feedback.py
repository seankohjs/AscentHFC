import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from typing import List
from functions import count_tokens, sanitize_text, load_feedback_data, load_files_in_date_range
from datetime import date, timedelta


# Load environment variables
load_dotenv(dotenv_path="config/.env")

st.title("Policy Feedback Analysis Chatbot")

# Sidebar with app explanation
st.sidebar.title("How to Use This Chatbot")
st.sidebar.write("""
This chatbot is designed to analyze user feedback on government policies and schemes. 
You can select the source of feedback data—either from chat history logs within a specified date range or directly from a policy feedback file.

Use the chatbot to ask questions or request analysis about the feedback data. You can inquire about:

- Overall sentiment or ratings given.
- Specific concerns or positive comments.
- Summaries of feedback across a particular date range (if using chat history).
- Patterns or trends in the collected feedback.
""")

# Data source selection
st.sidebar.header("Select Data Source")
data_source = st.sidebar.selectbox("Select Data Source", ["Chat History", "Policy Feedback"])

# Date range selection (only if analyzing chat history)
if data_source == "Chat History":
    st.sidebar.header("Select Date Range")
    start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=7))
    end_date = st.sidebar.date_input("End Date", date.today())


# Configure Gemini API using key from .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Generation config
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize model and chat session in session state
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )

if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []


# Load feedback data based on the selected data source and date range
if "feedback_data" not in st.session_state or st.session_state.get("selected_options", None) != (data_source, start_date if data_source == "Chat History" else None, end_date if data_source == "Chat History" else None):
    st.session_state.selected_options = (data_source, start_date if data_source == "Chat History" else None, end_date if data_source == "Chat History" else None)
    all_feedback_data = []
    if data_source == "Chat History":
        selected_files = load_files_in_date_range(start_date, end_date)
        for selected_file in selected_files:
            feedback_file_path = os.path.join("data", selected_file)
            all_feedback_data.extend(load_feedback_data(feedback_file_path))
    elif data_source == "Policy Feedback":
       feedback_file_path = os.path.join("data", "policy_feedback.txt")
       all_feedback_data = load_feedback_data(feedback_file_path)
    st.session_state.feedback_data = all_feedback_data
    
    # Calculate character count and approximate token count
    if st.session_state.feedback_data:
      combined_text = "\n".join(st.session_state.feedback_data)
      char_count = len(combined_text)
      approx_token_count = count_tokens(combined_text, st.session_state.model)
      st.session_state.char_count = char_count
      st.session_state.approx_token_count = approx_token_count
    else:
      st.session_state.char_count = 0
      st.session_state.approx_token_count = 0


# Set a flag for whether the first feedback prompt has been sent
if "first_prompt_sent" not in st.session_state:
    st.session_state.first_prompt_sent = False

# Display character count, token count and warning
st.sidebar.markdown("---")
st.sidebar.write(f"**Data Statistics:**")
st.sidebar.write(f"Approximate Characters: {st.session_state.char_count}")
st.sidebar.write(f"Approximate Tokens: {st.session_state.approx_token_count}")
if st.session_state.approx_token_count > 750000:
    st.sidebar.warning(
        "The token count is above 750,000. Please select a smaller data range or fewer files to avoid potential issues."
    )
st.sidebar.markdown("---")

# Display raw feedback data
with st.expander("View Raw Feedback"):
    if st.session_state.feedback_data:
        for line in st.session_state.feedback_data:
            st.write(line)
    else:
        st.write("No feedback data found")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your query here"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if st.session_state.feedback_data:
            # Join all feedback into a single string for context
            feedback_context = "\n".join(st.session_state.feedback_data)

            if not st.session_state.first_prompt_sent:
                enhanced_prompt = f"""You are a helpful assistant designed to analyze government policy feedback. Use the following feedback to answer the questions given. You should remember this feedback in future conversations.
                 Feedback:
                 {feedback_context}
                 
                 Instructions:
                 1. Analyze the given feedback to answer the specific questions given. 
                 2. If a rating has been provided, make note of it. 
                 3. Give an overview of the overall rating provided if applicable.
                 4. If the user asks to list items, list them and explain each item if necessary.
                 5. Structure your response with clear newlines to separate sentences and paragraphs for readability.
                 6. Use bullet points for lists to make information easy to digest.
                 7. Use headers where necessary to organize information effectively and enhance reader understanding.

                 User's question: {prompt}
                  """
                st.session_state.first_prompt_sent = True

            else:
                enhanced_prompt = f"""You are a helpful assistant designed to analyze government policy feedback. You should use the feedback given earlier to answer the following questions.

                 Instructions:
                 1. Analyze the feedback to answer the specific questions given. 
                 2. If a rating has been provided, make note of it. 
                 3. Give an overview of the overall rating provided if applicable.
                 4. If the user asks to list items, list them and explain each item if necessary.
                 5. Structure your response with clear newlines to separate sentences and paragraphs for readability.
                 6. Use bullet points for lists to make information easy to digest.
                 7. Use headers where necessary to organize information effectively and enhance reader understanding.

                 User's question: {prompt}
                  """

            # Count input tokens using the enhanced prompt
            input_tokens = count_tokens(enhanced_prompt, st.session_state.model)

            # Generate Gemini response with context
            with st.chat_message("assistant"):
                response = st.session_state.chat_session.send_message(enhanced_prompt)
                sanitized_response_text = sanitize_text(response.text)
                st.markdown(sanitized_response_text)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": sanitized_response_text})

        else:
            st.warning("No feedback data found, please try again after the user has provided feedback.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# Conditionally display the "End Chat" button in the sidebar
if len(st.session_state.messages) > 0:
    if st.sidebar.button("End Chat"):
        st.session_state.messages = []  # Clear chat history
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])  # Reset chat session
        st.session_state.first_prompt_sent = False  # reset the first prompt
        st.rerun()