import os
import google.generativeai as genai
import chromadb
from typing import List
from dotenv import load_dotenv
from functions import create_embedding, chunk_text, clean_text
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