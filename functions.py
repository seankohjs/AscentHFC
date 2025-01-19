import os
import google.generativeai as genai
import re
from typing import List
from datetime import date
import streamlit as st


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

def save_chat_history(messages: List[dict]):
    """Saves chat history to a file named with today's date."""
    # Ensure the 'chatHistory' folder exists
    history_dir = "chatHistory"
    os.makedirs(history_dir, exist_ok=True)
    
    today = date.today().strftime("%Y-%m-%d")
    file_path = os.path.join(history_dir, f"{today}.txt")

    try:
      with open(file_path, "a", encoding="utf-8") as f:
        for message in messages:
          f.write(f"{message['role']}: {message['content']}\n\n")
        f.write("-" * 40 + "\n\n") # Add a separator between conversations
      print(f"Chat history saved to: {file_path}")
    except Exception as e:
      st.error(f"Error saving chat history: {e}")