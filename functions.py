import os
import google.generativeai as genai
import re
from typing import List
from datetime import date, datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from textblob import TextBlob
from dotenv import load_dotenv
import pandas as pd


# Load environment variables
load_dotenv(dotenv_path="config/.env")

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

# Initialize model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# ---------- Text Processing Functions ----------

def clean_text(text: str) -> str:
    """Removes italics, special characters, and ensures plain text."""
    # Remove italics
    text = re.sub(r'[\*_]', '', text)  # Remove * and _ for italics
    
    # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Keep only ASCII characters and replace non-ascii with space
    
     # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def chunk_text(text: str) -> List[str]:
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def sanitize_text(text):
    """Escapes lone dollar signs, preserving spacing."""
    # Escape single dollar signs that are not part of LaTeX expressions, preserving spacing
    text = re.sub(r'(?<!\$)(?<!\\)\$(?!\$)', r'\$', text)
    return text
  
# ---------- Embedding and Token Functions ----------

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
    
# ---------- Chat History Functions ----------

def save_chat_history(user_message: str, assistant_message: str, new_session: bool, category: str = None, previous_assistant_message: str = None):
    """Saves the current user question and LLM reply to a file named with today's date, grouped by session and category."""
    
    history_dir = "data/chatHistory"
    os.makedirs(history_dir, exist_ok=True)

    normal_chat_dir = os.path.join(history_dir, "normalchat")
    os.makedirs(normal_chat_dir, exist_ok=True)
    
    feedback_dir = os.path.join(history_dir, "feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    
    pure_feedback_dir = os.path.join(history_dir, "purefeedback")
    os.makedirs(pure_feedback_dir, exist_ok=True)
    
    today = date.today().strftime("%Y-%m-%d")
    
    original_file_path = os.path.join(history_dir, f"{today}.txt")
    normal_chat_file_path = os.path.join(normal_chat_dir, f"normalchat_{today}.txt")
    feedback_file_path = os.path.join(feedback_dir, f"feedback_{today}.txt")
    pure_feedback_file_path = os.path.join(pure_feedback_dir, f"purefeedback_{today}.txt")
    
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
                if previous_assistant_message:
                    f.write(f"assistant: {previous_assistant_message}\n\n")
                f.write(f"user: {user_message}\t{now}\n\n")
            # Save pure feedback user message
            with open(pure_feedback_file_path, "a", encoding="utf-8") as f:
              f.write(f"user: {user_message}\t{now}\n\n")
              
        print(f"Chat history saved to: {original_file_path}")
        if category:
          print(f"Chat history saved to categorized file: {normal_chat_file_path if category == 'normalchat' else feedback_file_path if category == 'feedback' else None}")
          if category == 'feedback':
             print(f"Chat history saved to pure feedback file: {pure_feedback_file_path}")


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

# ---------- File Loading Functions ----------

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

# ---------- Feedback Analysis Functions ----------

def analyze_sentiment(text):
    """Analyzes the sentiment of the given text and returns a sentiment label."""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "positive"
    elif analysis.sentiment.polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def categorize_feedback_batch(texts, model):
  """Categorizes a list of feedback texts using Gemini API."""
  
  combined_texts = "\n".join([f"- {text}" for text in texts]) # Format a string for the prompt, each item on a new line
  classification_prompt = f"""You are a classification tool designed to categorize user feedback about government schemes into a category.

        Instructions:
        1. Analyze each user's feedback text below, and decide on a category for each of the user's feedback.
        2. The categories are as follows:
            "Scheme Specific Feedback"
            "General Feedback"
            "Chatbot Feedback"
        3. Return the category that best fits each of the user's feedback.
        4. The category of each feedback MUST be on a new line. Do not include any other text, only the category.
        4. The lines MUST match the same ordering as the feedback given below.

        User's feedback:
        {combined_texts}
        """
  try:
      chat_session = model.start_chat(history=[])
      response = chat_session.send_message(classification_prompt)
      cleaned_response = response.text.strip()
      
      results = []
      for line in cleaned_response.splitlines():
        results.append(line.strip())
      return results
  except Exception as e:
      print(f"Error classifying message: {e}")
      return ["Uncategorized"] * len(texts)

def process_feedback(file_path, batch_size, model):
  """Processes the pure feedback file, adds sentiment labels and categories in batches."""
  feedback_entries = []
  try:
      with open(file_path, 'r', encoding='utf-8') as f:
          batch = []
          for line in f:
            line = line.strip()
            if line: # Ensure the line is not empty
                parts = line.split("\t")
                if len(parts) == 2:  # Check if it is in the correct format.
                  text, timestamp = parts
                  batch.append((text, timestamp))
                  if len(batch) == batch_size:
                    texts, timestamps = zip(*batch)
                    sentiments = [analyze_sentiment(text) for text in texts]
                    categories= categorize_feedback_batch(texts, model)
                    for i, (text, timestamp) in enumerate(batch):
                       feedback_entries.append({
                            "text": text,
                            "timestamp": timestamp,
                            "sentiment": sentiments[i],
                            "category": categories[i],
                        })
                    batch = [] # reset the batch
          # Process any remaining items
          if batch:
            texts, timestamps = zip(*batch)
            sentiments = [analyze_sentiment(text) for text in texts]
            categories = categorize_feedback_batch(texts, model)
            for i, (text, timestamp) in enumerate(batch):
                feedback_entries.append({
                    "text": text,
                    "timestamp": timestamp,
                    "sentiment": sentiments[i],
                    "category": categories[i],
                })
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return []
  return feedback_entries

def get_all_feedback_data(start_date, end_date):
  """Loads and filters all the feedback data between the start and end dates"""
  data_dir = "data/chatHistory/purefeedback"
  feedback_files = []
  
  if os.path.exists(data_dir):
      for filename in os.listdir(data_dir):
          if filename.startswith("purefeedback_") and filename.endswith(".txt"):
              try:
                file_date_str = filename.replace("purefeedback_", "").replace(".txt", "")
                file_date = date.fromisoformat(file_date_str) # assumes file format YYYY-MM-DD
                if start_date <= file_date <= end_date:
                    feedback_files.append(os.path.join(data_dir, filename))
              except ValueError:
                  continue
  all_feedback = []
  for feedback_file in feedback_files:
      feedback_batch = process_feedback(feedback_file, 10, model)
      all_feedback.extend(feedback_batch)
  return pd.DataFrame(all_feedback)

def process_data(df):
    """Processes all the data for the dashboard."""
    # Metrics
    overall_sentiment = df['sentiment'].apply(lambda x: 1 if x == "positive" else -1 if x =="negative" else 0).mean()
    total_feedback = len(df)

    #Feedback Counts
    positive_feedback = len(df[df['sentiment'] == "positive"])
    negative_feedback = len(df[df['sentiment'] == "negative"])

     # Category Counts
    category_counts = df['category'].value_counts().to_dict()

    #Feedback Ratio
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    segments = {
        'Positive': sentiment_counts.get('positive', 0),
        'Neutral': sentiment_counts.get('neutral', 0),
        'Negative': sentiment_counts.get('negative', 0)
    }

    #Time Series
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df_daily = df.groupby('date').count().reset_index()
    df_daily.rename(columns={"text":"Feedback Count"}, inplace=True) #rename for plotting later
    df_daily['Date'] = pd.to_datetime(df_daily['date'])
    return overall_sentiment, total_feedback, positive_feedback, negative_feedback, category_counts, segments, df_daily

def summarize_feedback(df):
    combined_texts = "\n".join([f"- {text}" for text in df['text']])
    classification_prompt = f"""You are a helpful assistant that summarizes feedback for users.
         Instructions:
          1. Use the feedback from the users below to create a useful summarisation of feedback about government schemes.
          2. Provide an overall summary of the general feedback, as well as specific points about the different types of feedback.

        User Feedback:
          {combined_texts}
         """
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(classification_prompt)
        cleaned_response = response.text.strip()
        return cleaned_response
    except Exception as e:
        print(f"Error summarizing feedback: {e}")
        return "No summary available"