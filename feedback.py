import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from typing import List

# Load environment variables
load_dotenv()

def count_tokens(text: str, model: genai.GenerativeModel) -> int:
    """Counts tokens in a given text using the model's tokenizer."""
    try:
        response = model.count_tokens(text)
        return response.total_tokens
    except Exception as e:
      st.error(f"Error counting tokens: {e}")
      return 0

def sanitize_text(text):
    """Escapes lone dollar signs, preserving spacing."""
    # Escape single dollar signs that are not part of LaTeX expressions, preserving spacing
    text = re.sub(r'(?<!\$)(?<!\\)\$(?!\$)', r'\$', text)
    return text


def load_feedback_data(file_path: str) -> List[str]:
    """Loads feedback data from a file."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return []


st.title("Policy Feedback Analysis Chatbot")

# Sidebar with app explanation
st.sidebar.title("How to Use This Chatbot")
st.sidebar.write("""
This chatbot is designed to help analyze feedback about government policies and schemes.
You can interact with the chatbot to ask specific questions or request analysis
of the feedback data.
""")

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
    

# Load feedback data
feedback_file_path = "policy_feedback.txt"
feedback_data = load_feedback_data(feedback_file_path)

# Display raw feedback data
with st.expander("View Raw Feedback"):
    if feedback_data:
        for line in feedback_data:
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
        if feedback_data:
            # Join all feedback into a single string for context
            feedback_context = "\n".join(feedback_data)
            
            enhanced_prompt = f"""You are a helpful assistant designed to analyze government policy feedback. Use the following feedback to answer the questions given.
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
             """
            
             # Count input tokens using the enhanced prompt
            input_tokens = count_tokens(enhanced_prompt + prompt, st.session_state.model)
            
            # Generate Gemini response with context
            with st.chat_message("assistant"):
                response = st.session_state.chat_session.send_message(enhanced_prompt + prompt)
                sanitized_response_text = sanitize_text(response.text)
                st.markdown(sanitized_response_text)
                output_tokens = count_tokens(response.text, st.session_state.model)
                st.write(f"Total Tokens: {input_tokens + output_tokens}")

            
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
        st.rerun()