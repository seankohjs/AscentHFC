import os
import google.generativeai as genai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from typing import List
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection_name = "budgetinfo"  
collection = client.get_or_create_collection(name=collection_name)

def create_embedding(text: str) -> List[float]:
    """Create embedding for a single piece of text"""
    result = genai.embed_content(
        model="models/text-embedding-004",  # Make sure this model is available
        content=text
    )
    return result['embedding']

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

# Directory containing PDF documents
documents_folder = "documents"

print("Processing PDF documents...")
for filename in os.listdir(documents_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(documents_folder, filename)
        print(f"Processing: {filename}...")
        try:
            loader = PyPDFLoader(pdf_path)
            pdf_document = loader.load()

            for i, page in enumerate(pdf_document):
                cleaned_content = clean_text(page.page_content)
                chunks = chunk_text(cleaned_content)
                for j, chunk in enumerate(chunks):
                    embedding = create_embedding(chunk)
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        ids=[f"{filename}_page_{i}_chunk_{j}"],
                        metadatas=[{"source": filename, "page": i}]
                    )
            print(f"Finished processing: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Test queries for budget documents
test_queries = [
    "what are the requirements for the cost of living payments ?"

]


print("\nTesting queries...")
for query in test_queries:
    print(f"\nQuery: {query}")
    query_embedding = create_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["documents", "distances", "metadatas"]
    )
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0], results['distances'][0], results['metadatas'][0]
    )):
        similarity = 1 - (distance / 2)
        print(f"\nResult {i+1}:")
        print(f"Text: {doc.strip()}")
        print(f"Similarity Score: {similarity:.2f}")
        print(f"Source: {metadata['source']}, Page: {metadata['page']}")

print("\nTest completed!")