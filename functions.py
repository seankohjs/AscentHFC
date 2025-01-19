import os
import google.generativeai as genai
import re
from typing import List
from datetime import date, datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


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

def save_chat_history(user_message: str, assistant_message: str, new_session: bool):
    """Saves the current user question and LLM reply to a file named with today's date, grouped by session."""
    # Ensure the 'chatHistory' folder exists within the 'data' folder
    history_dir = "data/chatHistory"
    os.makedirs(history_dir, exist_ok=True)
    
    today = date.today().strftime("%Y-%m-%d")
    file_path = os.path.join(history_dir, f"{today}.txt")

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            if new_session:
              now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              f.write(f"Session started at: {now}\n")
              f.write("-" * 40 + "\n\n")
            f.write(f"user: {user_message}\n\n")
            f.write(f"assistant: {assistant_message}\n\n")
            f.write("-" * 40 + "\n\n")  # Add a separator between turns
        print(f"Chat history saved to: {file_path}")
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

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