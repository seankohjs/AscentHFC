import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import chromadb


# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY environment variable not set or found in .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "test_collection"
collection = client.get_collection(name=collection_name)

# Function to create embeddings using Gemini
def create_embedding(text: str) -> list[float]:
    """Create embedding for a single piece of text using Gemini"""
    result = genai.embed_content(
        model="models/text-embedding-004",  # Ensure this model is available
        content=text
    )
    return result['embedding']

# Create the model configuration *outside* the rerun scope
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

# Initialize chat_history *only once*
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize chat_session *only once* (outside the main logic)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat()
    st.session_state.chat_history.append({"role": "system", "content": "How can I assist you today?"})  # Initial system message
chat_session = st.session_state.chat_session  # Access it here

st.title("HFC Chat")

user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")

if user_input:
    try:
        # Step 1: Create embedding for the user's query
        query_embedding = create_embedding(user_input)

        # Step 2: Query ChromaDB for the most relevant documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Retrieve top 3 most relevant documents
            include=["documents", "metadatas"]
        )

        # Step 3: Combine the retrieved documents into a context for the LLM
        context = "Here are some relevant documents from the database:\n"
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"\nDocument {i+1} (Page {metadata['page']}):\n{doc}\n"

        # Step 4: Send the user's query along with the context to the LLM
        full_query = f"{context}\n\nUser's query: {user_input}"
        response = chat_session.send_message(full_query)

        # Step 5: Update chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.write(content)