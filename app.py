import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from typing import List
import re
import time
from functions import create_embedding, count_tokens, sanitize_text, save_chat_history, classify_message
from datetime import datetime


# Load environment variables
load_dotenv(dotenv_path="config/.env")

st.title("KiasuKaki")

# Sidebar with app explanation
st.sidebar.title("How Can I Help?")
st.sidebar.write("""
This chatbot is here to help you find information about various government schemes. 
Feel free to ask anything related to government schemes, such as how the government 
can help you, eligibility criteria, application processes, etc. 
""")

st.sidebar.markdown("---")  # Add a line separator

# Initialize consent state (set to True for default checked)
if "consent" not in st.session_state:
    st.session_state.consent = True

# Consent checkbox in sidebar (default value is set)
consent_checkbox = st.sidebar.checkbox(
    "I consent to my conversation being recorded", value=st.session_state.consent
)

# Update consent state based on checkbox
st.session_state.consent = consent_checkbox

# Conditional Disclaimer
if st.session_state.consent:
    st.sidebar.markdown("""
        **Disclaimer:** This conversation **is being recorded** to help improve government 
        policies and schemes in the future.
        """)
else:
    st.sidebar.markdown("""
        **Disclaimer:** This conversation **is not being recorded**. Please note that without your consent,
        your conversation will not be saved to help improve government policies and schemes.
        """)

# Sidebar with feedback form
# with st.sidebar.form(key="policy_feedback_form", clear_on_submit=True):
#     st.write("Tell us what you think about the government schemes:")
    
#     feedback = st.text_area(
#         "What are your thoughts on the policies/schemes discussed?",
#         placeholder="Enter your feedback here...",
#         height=150,
#     )
    
#     rating = st.slider("On a scale of 1 to 5, how helpful do you find these schemes?", 1, 5, 3)
    
#     submit_feedback = st.form_submit_button("Submit Feedback")

#     if submit_feedback:
#         if feedback:  # Check if feedback is not empty
#             # Get current date and time
#             now = datetime.now()
#             timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

#             # Save feedback to a file (append mode)
#             with open("data/policy_feedback.txt", "a") as f:
#                 f.write(f"Timestamp: {timestamp}\n")
#                 f.write(f"Feedback: {feedback}\nRating: {rating}\n\n")  # Append feedback with a separator
#             st.sidebar.success("Thank you for your feedback! It's very valuable.")
#         else:
#             st.sidebar.warning("Please provide some feedback before submitting.")


# Initialize ChromaDB with path to local database
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("budgetinfo")

# Configure Gemini API using key from .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Generation config
generation_config = {
    "temperature": 1,
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

# Initialize token counters in session state
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
  st.session_state.total_output_tokens = 0

# Set a flag for whether it's a new session
if "new_session" not in st.session_state:
    st.session_state.new_session = True

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type something"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        # Create embedding for the query
        query_embedding = create_embedding(prompt)
        
        # Query ChromaDB using the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=60,  # Get top 5 most relevant documents
            include=["documents","metadatas"]
        )
        
        # Prepare context from retrieved documents
        context_docs = results['documents'][0]  # List of retrieved document texts
        context_metadata = results['metadatas'][0]  # list of retrieved metadatas

        # Format the chat history to include in prompt
        chat_history = ""
        for message in st.session_state.messages[:-1]:
            if message["role"] == "user":
                chat_history += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
               chat_history += f"Assistant: {message['content']}\n"
        
        # Create enhanced prompt with context
        enhanced_prompt = f"""You are a helpful and informative assistant chatbot designed to provide citizens with information about government schemes. Your goal is to provide clear, accurate, and well-formatted information based on the documents provided and the previous conversation history. You will add a formatted feedback line to the end of your responses, *unless* the user has asked a question about providing feedback.

        Context from Budget 2024 documents about government schemes:
        {' '.join(context_docs)}
        
        Previous Conversation:
        {chat_history}

        User's question: {prompt}

        Instructions:
        1. Base your response ONLY on the provided context from the Budget 2024 documents and the previous conversation. Avoid introducing external knowledge or assumptions.
        2. Provide a clear and concise answer that is easy to understand for the average citizen. Do not include italicized words, bold text, or any special formatting unless explicitly required by the user. The output text should contain only standard text characters.
        3. If the user's question is *only* a simple greeting (e.g., "hi", "hello", "good morning"), acknowledge the greeting politely, and state that you are a chatbot designed to provide information about government schemes, and then ask how you can assist them. Do *not* provide a list of schemes. If the user's question is not a simple greeting, but is vague or ambiguous, or if the question is related to the topic but the context is insufficient to answer it directly, **ask a specific clarifying question** to help you better understand the user's needs, *before* stating that you do not have enough information to answer specifically from the Budget 2024 documents. Do not state "I do not have enough information" without first making an attempt to understand the user's needs. If you are able to provide some relevant context even if you can not fully answer the question, provide that relevant context.
        4. When applicable, include specific details like dates, amounts, or specific scheme names from the context to be most accurate. Ensure that numerical ranges are formatted correctly with spaces (e.g., "200 to 400"), and there is a space after any number and before any word. Remove any extraneous text, such as the names of schemes or documents, that may be next to each requirement if they do not add clarity.
        5. Avoid carrying over formatting from source documents that may include italics, bold text, or other stylistic choices unless they are necessary for clarity.
        6. If the provided context has multiple options that may answer the question, provide all options, and explain all of them clearly.
        7. If the information from the context may be confusing or has multiple meanings, explain each option clearly, without making a specific assumption.
        8. Do not generate or include information not found in the provided document.
        9. Prioritize clarity and accuracy in your responses. If there are discrepancies or outdated information from blog posts or less reliable sources, prioritize information aligned with official government documents when available.
        10. Structure your response with clear newlines to separate sentences and paragraphs for readability.
        11. Use bullet points for lists to make information easy to digest.
        12. Use headers where necessary to organize information effectively and enhance reader understanding.
        13. Sanitize the output to ensure that text is clean and consistent, avoiding any carryover of special formatting or symbols from source documents, and that text is spaced correctly with numbers.
        14. The response should only have standard text characters, no html characters or special characters.
        15. If you do not have enough information to provide an answer specifically from the Budget 2024 documents, ask a clarifying question so that the user can be more specific to give you enough information to answer their question.
        16. If the user asks where they can give feedback, tell them that they can give feedback directly to the chatbot, and they can also visit official government websites or use official government feedback channels.
        17. If the user provides feedback, acknowledge and thank them for it, using an elegant tone. If the feedback also includes a question, respond to the question. Use the chat history to determine if the user is providing feedback.
        18. *Unless* the user has asked a question about providing feedback, add a new line, and then add the following message as a separate line at the end of your response: '\n\n---\n**If you have any feedback, you may provide it directly to this bot, visit official government websites, or use government feedback channels.**'
        """
         # Count input tokens using the enhanced prompt
        input_tokens = count_tokens(enhanced_prompt, st.session_state.model)
        st.session_state.total_input_tokens += input_tokens
        
        # Generate Gemini response with context
        with st.chat_message("assistant"):

            # Get response from Gemini
            response = st.session_state.chat_session.send_message(enhanced_prompt)

            sanitized_response_text = sanitize_text(response.text)
            
            # Check if the user asked how to give feedback, and don't add the footer if they did
            # REMOVE THIS ENTIRE LINE
            #if "give feedback" not in prompt.lower():
            #  sanitized_response_text += ' \n\n If you have any feedback, you may provide it directly to this bot, visit official government websites, or use government feedback channels.'

            st.markdown(sanitized_response_text)
            
            # Count response tokens
            output_tokens = count_tokens(response.text, st.session_state.model)
            st.session_state.total_output_tokens += output_tokens
        
        # Classify the message
        chat_history_for_classification = ""
        for message in st.session_state.messages:
            if message["role"] == "user":
                chat_history_for_classification += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
               chat_history_for_classification += f"Assistant: {message['content']}\n"
        
        classification = classify_message(chat_history_for_classification, prompt, st.session_state.model)
        
        # Get the previous assistant message if it exists
        previous_assistant_message = None
        for message in reversed(st.session_state.messages):
            if message["role"] == "assistant":
                previous_assistant_message = message["content"]
                break
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": sanitized_response_text})
        
        # Save the entire chat history for this interaction only if the user gave consent
        if st.session_state.consent:
            save_chat_history(prompt, sanitized_response_text, st.session_state.new_session, category=classification, previous_assistant_message = previous_assistant_message)
        
        # Reset new_session to False after first message
        st.session_state.new_session = False
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Conditionally display the "End Chat" button in the sidebar
if len(st.session_state.messages) > 0:
    if st.sidebar.button("End Chat"):
        st.session_state.messages = []  # Clear chat history
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])  # Reset chat session
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.new_session = True  # Reset for next session
        st.rerun()