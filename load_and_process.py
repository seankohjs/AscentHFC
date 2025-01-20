import pandas as pd
from datetime import datetime
import os
import google.generativeai as genai
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import date

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
    df = df.set_index('timestamp') #Make sure that your timestamp column is a datetime object.
    df_monthly = df.resample('M').count() #Group by month
    df_monthly.rename(columns={"text":"Feedback Count"}, inplace=True) #rename for plotting later
    df_monthly["Date"] = df_monthly.index # add in a date column
    return overall_sentiment, total_feedback, positive_feedback, negative_feedback, category_counts, segments, df_monthly

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