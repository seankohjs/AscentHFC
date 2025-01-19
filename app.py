import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from typing import List
import re
import time
from functions import create_embedding, count_tokens, sanitize_text, save_chat_history

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
st.sidebar.write("""
**Disclaimer:** This conversation will be recorded to help improve government 
policies and schemes in the future.
""")


# Sidebar with feedback form
with st.sidebar.form(key="policy_feedback_form", clear_on_submit=True):
    st.write("Tell us what you think about the government schemes:")
    
    feedback = st.text_area(
        "What are your thoughts on the policies/schemes discussed?",
        placeholder="Enter your feedback here...",
        height=150,
    )
    
    rating = st.slider("On a scale of 1 to 5, how helpful do you find these schemes?", 1, 5, 3)
    
    submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        if feedback:  # Check if feedback is not empty
            # Save feedback to a file (append mode)
            with open("data/policy_feedback.txt", "a") as f:
                f.write(f"Feedback: {feedback}\nRating: {rating}\n\n")  # Append feedback with a separator
            st.sidebar.success("Thank you for your feedback! It's very valuable.")
        else:
            st.sidebar.warning("Please provide some feedback before submitting.")

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
        
        # Create enhanced prompt with context
        enhanced_prompt = f"""You are a helpful and informative assistant chatbot designed to provide citizens with information about government schemes. Your goal is to provide clear, accurate, and well-formatted information based on the documents provided.

        Context from Budget 2024 documents about government schemes:
        {' '.join(context_docs)}

        User's question: {prompt}

        Instructions:
        1. Base your response ONLY on the provided context from the Budget 2024 documents. Avoid introducing external knowledge or assumptions.
        2. Provide a clear and concise answer that is easy to understand for the average citizen. Do not include italicized words, bold text, or any special formatting unless explicitly required by the user. The output text should contain only standard text characters.
        3. If the provided context from the Budget 2024 documents does not fully answer the question, or if the user's question is ambiguous, state that you do not have enough information to answer specifically from the Budget 2024 documents, and that the user may need to consult official authorities, or ask additional clarifying questions. Then, **ask a specific clarifying question** to help you better understand the user's needs.
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
        """
        
         # Count input tokens using the enhanced prompt
        input_tokens = count_tokens(enhanced_prompt, st.session_state.model)
        st.session_state.total_input_tokens += input_tokens
        
        # Generate Gemini response with context
        with st.chat_message("assistant"):

            # Get response from Gemini
            response = st.session_state.chat_session.send_message(enhanced_prompt)

            sanitized_response_text = sanitize_text(response.text)

            st.markdown(sanitized_response_text)
            
            # Count response tokens
            output_tokens = count_tokens(response.text, st.session_state.model)
            st.session_state.total_output_tokens += output_tokens
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": sanitized_response_text})
        
        # Save the entire chat history for this interaction
        save_chat_history(prompt, sanitized_response_text, st.session_state.new_session)
        
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