import os
import google.generativeai as genai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB with new client syntax
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection_name = "test_collection"
collection = client.get_or_create_collection(name=collection_name)

def create_embedding(text: str) -> List[float]:
    """Create embedding for a single piece of text"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']

def chunk_text(text: str) -> List[str]:
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

# Test documents
documents = [
    """
    Machine Learning is a subset of artificial intelligence that focuses on developing 
    systems that can learn from and make decisions based on data. It involves training 
    models using algorithms to recognize patterns and make predictions.
    """,
    """
    Python is a high-level programming language known for its simplicity and readability. 
    It's widely used in data science, web development, and automation. Python's extensive 
    library ecosystem makes it popular among developers.
    """,
    """
    Vector databases are specialized database systems designed to store and query vector 
    embeddings efficiently. They are crucial for implementing semantic search and 
    similarity matching in modern applications.
    """
]

# Process and add documents
print("Processing documents...")
for i, doc in enumerate(documents):
    chunks = chunk_text(doc)
    for j, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"doc_{i}_chunk_{j}"],
            metadatas=[{"source": f"document_{i}"}]
        )

# Test queries
test_queries = [
    "What is machine learning?",
    "Tell me about Python programming",
    "How do vector databases work?",
    "What is data science?",  # This is interesting as it's related but not directly mentioned
]

print("\nTesting queries...")
for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Create embedding for query
    query_embedding = create_embedding(query)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,  # Get top 2 results
        include=["documents", "distances", "metadatas"]
    )
    
    # Print results
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        similarity = 1 - (distance / 2)  # Convert distance to similarity score
        print(f"\nResult {i+1}:")
        print(f"Text: {doc.strip()}")
        print(f"Similarity Score: {similarity:.2f}")
        print(f"Source: {metadata['source']}")

print("\nTest completed!")