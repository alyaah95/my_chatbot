# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS to handle cross-origin requests
import nltk
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes, allowing your frontend to connect
CORS(app) 

# --- NLTK Download (Run once) ---
# Ensure the 'punkt' tokenizer data is downloaded for English.
# This explicitly tells NLTK to download the standard English tokenizer model.
# This will only download if not already present.
nltk.download('punkt')

# --- RAG Model Components ---
# Global variables to store the loaded RAG model components
rag_index = None
rag_vectorizer = None
rag_chunks = None

# --- 1. Load Documents ---
def load_documents(folder_path):
    """
    Loads text content from all .txt files within the specified folder.
    Args:
        folder_path (str): The path to the folder containing the documents.
    Returns:
        str: A single string containing the combined text from all documents.
    """
    documents = []
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        print("Please create this folder and place your .txt files inside it.")
        return "" 

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    if not documents:
        print(f"Warning: No .txt files found in '{folder_path}' or files are empty.")
        return ""
    
    return " ".join(documents)

# --- 2. Chunking (Using Regular Expressions) ---
def chunk_text(text):
    """
    Splits the given text into sentences using regular expressions.
    This avoids reliance on NLTK's built-in tokenizers which caused issues.
    Args:
        text (str): The input text to be chunked.
    Returns:
        list: A list of sentences (chunks).
    """
    if not text:
        return []
    # This regular expression splits text by common sentence-ending punctuation
    # (. ! ?) followed by one or more spaces and then an uppercase letter (start of next sentence).
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

# --- 3. Embed & Store ---
def create_vector_store(chunks):
    """
    Converts text chunks into numerical vectors using TF-IDF and stores them in a FAISS index.
    Args:
        chunks (list): A list of text chunks (sentences).
    Returns:
        tuple: A tuple containing the FAISS index and the trained TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray().astype('float32')
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index, vectorizer

# --- 4. Retrieval ---
def retrieve_chunks(query, index, vectorizer, chunks, k=3):
    """
    Retrieves the most relevant text chunks from the vector store based on a user query.
    Args:
        query (str): The user's question.
        index (faiss.Index): The FAISS index containing document vectors.
        vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The trained vectorizer.
        chunks (list): The original list of text chunks.
        k (int): The number of top relevant chunks to retrieve.
    Returns:
        list: A list of the most relevant text chunks.
    """
    if not query:
        return []
    
    # Transform the user's query into a vector using the SAME vectorizer
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    
    # Search the FAISS index for the 'k' most similar vectors to the query vector.
    distances, indices = index.search(query_vector, k)
    
    # Retrieve the actual text chunks using the obtained indices
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    return retrieved_chunks

# --- 5. Generation ---
def generate_answer(query, context):
    """
    Generates a simple answer based on the retrieved context.
    (This is a basic placeholder as no full LLM is used).
    Args:
        query (str): The user's original question.
        context (str): The combined retrieved text chunks.
    Returns:
        str: A simple answer string.
    """
    if not context:
        return f"Sorry, I couldn't find relevant information for your question: '{query}'."
    return f"Based on the available information regarding '{query}':\n{context}"

# --- API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"answer": "Please provide a question."}), 400

    global rag_index, rag_vectorizer, rag_chunks

    # Ensure the RAG model is loaded before processing queries
    if rag_index is None or rag_vectorizer is None or rag_chunks is None:
        return jsonify({"answer": "RAG model not initialized. Please restart the server."}), 500

    try:
        retrieved_chunks = retrieve_chunks(query, rag_index, rag_vectorizer, rag_chunks)
        context = " ".join(retrieved_chunks)
        answer = generate_answer(query, context)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during retrieval or generation: {e}")
        return jsonify({"answer": "An error occurred while processing your request."}), 500

# --- Model Initialization on App Startup ---
# This block runs once when the Flask app starts
with app.app_context():
    print("Loading RAG model (documents, chunking, embedding)...")
    documents_folder = "documents" # Specify the folder name
    
    corpus = load_documents(documents_folder)
    
    if not corpus:
        print("Error: No documents loaded. The RAG model will not be able to answer questions.")
    else:
        chunks = chunk_text(corpus)
        if not chunks:
            print("Error: No valid sentences were found after chunking. RAG model will not function.")
        else:
            rag_index, rag_vectorizer = create_vector_store(chunks)
            rag_chunks = chunks
            print("RAG model initialized successfully!")

if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' makes it accessible from other devices on your local network
    # port=5000 is the default Flask port
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True is good for development
