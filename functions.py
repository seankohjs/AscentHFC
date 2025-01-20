import os
import google.generativeai as genai
import re
from typing import List
from datetime import date, datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def create_embedding(text: str) -> List[float]:
    """Create embedding for a single piece of text"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']

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

def save_chat_history(user_message: str, assistant_message: str, new_session: bool, category: str = None):
    """Saves the current user question and LLM reply to a file named with today's date, grouped by session and category."""
    
    history_dir = "data/chatHistory"
    os.makedirs(history_dir, exist_ok=True)

    normal_chat_dir = os.path.join(history_dir, "normalchat")
    os.makedirs(normal_chat_dir, exist_ok=True)
    
    feedback_dir = os.path.join(history_dir, "feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    
    today = date.today().strftime("%Y-%m-%d")
    
    original_file_path = os.path.join(history_dir, f"{today}.txt")
    normal_chat_file_path = os.path.join(normal_chat_dir, f"normalchat_{today}.txt")
    feedback_file_path = os.path.join(feedback_dir, f"feedback_{today}.txt")
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Save the original chat history (user + assistant)
        with open(original_file_path, "a", encoding="utf-8") as f:
            if new_session:
                f.write(f"Session started at: {now}\n")
                f.write("-" * 40 + "\n\n")
            f.write(f"user: {user_message}\t{now}\n\n")
            f.write(f"assistant: {assistant_message}\n\n")
            f.write("-" * 40 + "\n\n")  # Add a separator between turns

        # Save user message to categorized file
        if category == "normalchat":
            with open(normal_chat_file_path, "a", encoding="utf-8") as f:
                f.write(f"user: {user_message}\t{now}\n\n")
        elif category == "feedback":
            with open(feedback_file_path, "a", encoding="utf-8") as f:
              f.write(f"user: {user_message}\t{now}\n\n")

        print(f"Chat history saved to: {original_file_path}")
        if category:
          print(f"Chat history saved to categorized file: {normal_chat_file_path if category == 'normalchat' else feedback_file_path}")


    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def classify_message(chat_history: str, current_message: str, model: genai.GenerativeModel) -> str:
    """Classifies the current message as 'normalchat' or 'feedback' using Gemini."""
    classification_prompt = f"""You are a classification tool designed to categorize user messages.

        Instructions:
        1.  Analyze the current message and previous chat history provided by the user.
        2.  Determine whether the current message is a 'normalchat' message, in which the user is asking a question or a query about the topic, or a 'feedback' message, in which the user is providing feedback.
        3.  Return ONLY one of two strings: 'normalchat' or 'feedback' as the classification.
        4.  Do NOT include any additional text, just the classification.

        Previous Chat History:
        {chat_history}

        Current Message: {current_message}
        """
    try:
        # Start a new chat session for classification
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(classification_prompt)
        cleaned_response = response.text.strip().lower()

        if "normalchat" in cleaned_response:
            return "normalchat"
        elif "feedback" in cleaned_response:
            return "feedback"
        else:
            return "normalchat" # return default classification if not clear
    
    except Exception as e:
        st.error(f"Error classifying message: {e}")
        return "normalchat" # default case if it errors out

def chunk_text(text: str) -> List[str]:
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def clean_text(text: str) -> str:
    """Removes italics, special characters, and ensures plain text."""
    # Remove italics
    text = re.sub(r'[\*_]', '', text)  # Remove * and _ for italics
    
    # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Keep only ASCII characters and replace non-ascii with space
    
     # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def load_feedback_data(file_path: str) -> List[str]:
    """Loads feedback data from a file."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return []
    
# Function to load files within the selected date range
def load_files_in_date_range(start_date, end_date):
    data_dir = "data"
    chat_history_dir = "data/chatHistory"
    feedback_files = []
    
    if os.path.exists(chat_history_dir):
      for filename in os.listdir(chat_history_dir):
          if filename.endswith(".txt"):
              try:
                  file_date = date.fromisoformat(filename.replace(".txt", ""))  # assumes file format YYYY-MM-DD
                  if start_date <= file_date <= end_date:
                      feedback_files.append(f"chatHistory/{filename}")
              except ValueError:
                  continue  # skip if filename doesn't match date format

    return feedback_files