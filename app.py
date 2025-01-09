import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY environment variable not set or found in .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)


def get_chat_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

chat_history = get_chat_history()
chat_session = model.start_chat()

st.title("Gemini Chat")

user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")

if user_input:
    try:
        response = chat_session.send_message(user_input)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response.text}) # Access the text

        st.session_state.chat_history = chat_history

    except Exception as e:
        st.error(f"An error occurred: {e}")


for message in chat_history:
    role = message["role"] # More readable
    content = message["content"]
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    elif role == "assistant": # Correctly check role to access message
        with st.chat_message("assistant"):
            st.write(content)