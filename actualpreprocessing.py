import os
import google.generativeai as genai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection_name = "test_collection"
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
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

# Load PDF
pdf_path = "./processtest.pdf"  # Replace with your PDF file path
loader = PyPDFLoader(pdf_path)
pdf_document = loader.load()

# Process and add documents from the PDF
print("Processing PDF...")
for i, page in enumerate(pdf_document):
    chunks = chunk_text(page.page_content)
    for j, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"page_{i}_chunk_{j}"],
            metadatas=[{"source": f"processtest.pdf", "page": i}]
        )

# Test queries (Adapt these to your PDF content)
test_queries = [
    "What is the main topic of this document?",
    "Find information about key concepts",
    "Summarize the findings", # example queries
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