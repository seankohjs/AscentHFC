import os
import google.generativeai as genai
from textblob import TextBlob
from dotenv import load_dotenv

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
  classification_prompt = f"""You are a classification tool designed to categorize user feedback about government schemes into a category and a subcategory.

        Instructions:
        1. Analyze each user's feedback text below, and decide on a category and subcategory for each of the user's feedback.
        2. The categories and subcategories are as follows:
            "Scheme Specific Feedback":
                "CDC Vouchers":
                "U-Save Rebates":
                "SkillsFuture Credit":
                "Cost of Living Payment":
                "Housing Schemes":
                "Workfare Income Supplement":
                 "Overseas Humanitarian Assistance Tax Deduction Scheme":
            "General Feedback":
                "Clarity and Accessibility of Information":
                "Adequacy of Support":
                 "Overall Impact of Schemes":
            "Chatbot Feedback":
                "Usefulness and Helpfulness":
                "Specific Bot Requests":
        3. Return a category and a subcategory that best fits each of the user's feedback.
        4. The category and subcategory of each feedback MUST be on the same line, and separated by a comma, eg, 'Chatbot Feedback, Usefulness and Helpfulness'. Do not include any other text. Do NOT include any additional text other than the category and subcategory on each line. The lines MUST match the same ordering as the feedback given below.

        User's feedback:
        {combined_texts}
        """
  try:
      chat_session = model.start_chat(history=[])
      response = chat_session.send_message(classification_prompt)
      cleaned_response = response.text.strip()
      
      results = []
      for line in cleaned_response.splitlines():
        parts = line.split(',')
        if len(parts) == 2:
          results.append((parts[0].strip(), parts[1].strip()))
        else:
           results.append(("Uncategorized", "N/A"))
      return results
  except Exception as e:
      print(f"Error classifying message: {e}")
      return [("Uncategorized", "N/A")] * len(texts)

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
                    categories_and_subcategories = categorize_feedback_batch(texts, model)
                    for i, (text, timestamp) in enumerate(batch):
                       category, subcategory = categories_and_subcategories[i]
                       feedback_entries.append({
                            "text": text,
                            "timestamp": timestamp,
                            "sentiment": sentiments[i],
                            "category": category,
                            "subcategory": subcategory
                        })
                    batch = [] # reset the batch
          # Process any remaining items
          if batch:
            texts, timestamps = zip(*batch)
            sentiments = [analyze_sentiment(text) for text in texts]
            categories_and_subcategories = categorize_feedback_batch(texts, model)
            for i, (text, timestamp) in enumerate(batch):
                category, subcategory = categories_and_subcategories[i]
                feedback_entries.append({
                    "text": text,
                    "timestamp": timestamp,
                    "sentiment": sentiments[i],
                    "category": category,
                    "subcategory": subcategory
                })
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return []
  return feedback_entries

if __name__ == "__main__":
    file_path = "/Users/seankohjs/Documents/GitHub/AscentHFC/data/chatHistory/purefeedback/purefeedback_2025-01-21.txt"  # Specify the file path
    batch_size = 5 # Set the batch size
    feedback_with_categories = process_feedback(file_path, batch_size, model)

    if feedback_with_categories:
        for entry in feedback_with_categories:
            print(f"Text: {entry['text']}")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Sentiment: {entry['sentiment']}")
            print(f"Category: {entry['category']}")
            print(f"Subcategory: {entry['subcategory']}\n")