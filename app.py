import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from typing import List

# Load environment variables
load_dotenv()

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

st.title("Government Scheme Assistance Bot")

# Sidebar with app explanation
st.sidebar.title("About This App")
st.sidebar.write("""
This app is designed to assist users in navigating and understanding various government schemes. 
It uses advanced AI to provide relevant information based on user queries. The app integrates 
with a database of documents and uses natural language processing to deliver accurate and 
context-aware responses.
""")

# Feedback form in the sidebar
with st.sidebar.form(key="feedback_form"):
    st.write("We'd love to hear your feedback!")
    feedback = st.text_area(
        "How was your experience with the bot?",
        placeholder="Enter your feedback here...",
        height=100,
    )
    submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        if feedback:  # Check if feedback is not empty
            # Save feedback to a file (append mode)
            with open("feedback.txt", "a") as f:
                f.write(f"Feedback: {feedback}\n\n")  # Append feedback with a separator
            st.sidebar.success("Thank you for your feedback! We appreciate it.")
        else:
            st.sidebar.warning("Please enter some feedback before submitting.")

# Initialize ChromaDB with path to local database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("test_collection")

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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
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
            n_results=5  # Get top 5 most relevant documents
        )
        
        # Prepare context from retrieved documents
        context_docs = results['documents'][0]  # List of retrieved document texts
        
        # Create enhanced prompt with context
        enhanced_prompt = f"""Based on the following context and the user's question, provide a relevant answer.

Context from documents:
{' '.join(context_docs)}

User's question: {prompt}

Please provide a response that incorporates relevant information from the context."""
        
         # Count input tokens using the enhanced prompt
        input_tokens = count_tokens(enhanced_prompt, st.session_state.model)
        st.session_state.total_input_tokens += input_tokens
        
        # Generate Gemini response with context
        with st.chat_message("assistant"):
            # Show retrieved documents in expander (for debugging)
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(context_docs, 1):
                    st.write(f"Document {i}:", doc)
            
            # Get response from Gemini
            response = st.session_state.chat_session.send_message(enhanced_prompt)
            st.markdown(response.text)
            
            # Count response tokens
            output_tokens = count_tokens(response.text, st.session_state.model)
            st.session_state.total_output_tokens += output_tokens
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Conditionally display the "End Chat" button in the sidebar
if len(st.session_state.messages) > 0:
    if st.sidebar.button("End Chat"):
        st.session_state.messages = []  # Clear chat history
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])  # Reset chat session
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun() # plan to change this to ask for feedback

# Display token counts in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Token Usage")
st.sidebar.write(f"Total Input Tokens: {st.session_state.total_input_tokens}")
st.sidebar.write(f"Total Output Tokens: {st.session_state.total_output_tokens}")
st.sidebar.write(f"Total Tokens: {st.session_state.total_input_tokens + st.session_state.total_output_tokens}")