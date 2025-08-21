# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import faiss
import numpy as np
import os
import re

# Import necessary libraries for the Small Language Models
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer 

# Import OpenAI library
import openai

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# --- OpenAI API Key Configuration ---
# IMPORTANT: Replace "YOUR_OPENAI_API_KEY_HERE" with your actual OpenAI API Key.
# For better security in a real application, store this in an environment variable.
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE" 
openai.api_key = OPENAI_API_KEY

# --- NLTK Download (Run once) ---
nltk.download('punkt')

# --- RAG Model Components ---
rag_index = None
rag_embedding_model = None 
rag_chunks = None

# --- SLM for Summarization ---
slm_model = None
slm_tokenizer = None

# --- RAG Functions ---
def load_documents(folder_path):
    """
    Loads text content from all .txt files within the specified folder.
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

def chunk_text(text):
    """
    Splits the given text into sentences using regular expressions.
    """
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def create_vector_store(chunks, embedding_model): 
    """
    Converts text chunks into numerical vectors using SentenceTransformer and stores them in a FAISS index.
    """
    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')
    
    d = chunk_embeddings.shape[1] 
    index = faiss.IndexFlatL2(d)
    index.add(chunk_embeddings)
    
    return index

def retrieve_chunks(query, index, embedding_model, chunks, k=5, relevance_threshold=0.8): 
    """
    Retrieves the most relevant text chunks and their distances from the vector store based on a user query.
    Filters chunks based on a relevance threshold.
    Returns:
        tuple: (list of filtered retrieved chunk texts, list of their corresponding L2 distances)
    """
    if not query:
        return [], np.array([])
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    
    distances, indices = index.search(query_embedding, k)
    
    if indices.size == 0:
        return [], np.array([])
        
    filtered_chunks_text = []
    filtered_distances = []

    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        distance = distances[0][i]
        # Only include chunks that are within the relevance threshold
        # Lower distance = higher similarity
        if distance < relevance_threshold: 
            chunk_text = chunks[chunk_index]
            if chunk_text.strip(): # Ensure chunk is not empty after stripping
                filtered_chunks_text.append(chunk_text)
                filtered_distances.append(distance)
        else:
            # Once a chunk is too dissimilar, stop looking further (assuming sorted by distance)
            break 
            
    return filtered_chunks_text, np.array(filtered_distances)

def generate_answer(query, context):
    """
    Generates a simple answer based on the retrieved context.
    """
    if not context:
        return f"Sorry, I couldn't find relevant information for your question: '{query}'."
    # You can enhance this by asking a small LLM to synthesize the answer
    # from the context, but for now, we concatenate.
    return f"Based on the available information regarding '{query}':\n{context}"

# --- OpenAI API Call Function ---
def call_openai_llm(prompt_text, model="gpt-3.5-turbo", max_tokens=200, temperature=0.7):
    """
    Calls the OpenAI Chat Completions API.
    """
    try:
        if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY_HERE":
            print("OpenAI API Key is not set or is default. Cannot call OpenAI.")
            return "OpenAI API Key is not configured."

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide concise and accurate answers."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"OpenAI API error occurred: {e}")
        # Return a more informative error message to the user
        return f"An OpenAI API error occurred: {e.status_code if hasattr(e, 'status_code') else 'N/A'}. Details: {e.response.json().get('error', {}).get('message', 'No specific message.') if hasattr(e, 'response') and hasattr(e.response, 'json') else str(e)}"
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI API call: {e}")
        return f"An unexpected error occurred with OpenAI: {e}"


# --- API Endpoints ---
@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint for asking a question and getting a RAG-based answer with OpenAI fallback."""
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"answer": "Please provide a question.", "source": "User Error"}), 400

    global rag_index, rag_embedding_model, rag_chunks
    if rag_index is None or rag_embedding_model is None or rag_chunks is None:
        return jsonify({"answer": "RAG model not initialized. Please restart the server.", "source": "Server Error"}), 500

    # Define a relevance threshold for L2 distance from SentenceTransformer embeddings.
    # Lower distance means higher similarity. If the smallest distance is above this, RAG likely found no relevant info.
    # This value needs to be tuned. A distance of 0.8 is quite dissimilar for MiniLM.
    # We are now filtering inside retrieve_chunks based on this.
    RELEVANCE_THRESHOLD = 0.8 # Adjust this value (e.g., 0.6, 0.7, 0.9) based on testing

    rag_answer_source = "Internal Documents"
    answer_text = ""
    
    try:
        # Retrieve chunks and their distances, filtered by the relevance threshold
        retrieved_chunks, distances = retrieve_chunks(query, rag_index, rag_embedding_model, rag_chunks, k=5, relevance_threshold=RELEVANCE_THRESHOLD) 
        
        # Check if any relevant chunks were found after filtering
        if retrieved_chunks:
            context_from_rag = " ".join(retrieved_chunks)
            answer_text = generate_answer(query, context_from_rag)
            # Log the closest distance for debugging/tuning the threshold
            print(f"RAG found relevant context (closest distance: {distances[0]:.3f}) for query: '{query}'.")
        else:
            print(f"RAG found no relevant chunks (closest distance was too high or no chunks) for query: '{query}'. Falling back to OpenAI...")
            rag_answer_source = "OpenAI (Fallback)"
            openai_prompt = f"Answer the following question concisely: {query}"
            answer_text = call_openai_llm(openai_prompt)
            if not answer_text or "API Key is not configured." in answer_text: 
                answer_text = "Sorry, I couldn't find a relevant answer in my documents or with the fallback AI. Please check server logs for more details."
                rag_answer_source = "No Answer"
        
        return jsonify({"answer": answer_text, "source": rag_answer_source})

    except Exception as e:
        print(f"An unexpected error occurred in /ask endpoint: {e}")
        return jsonify({"answer": f"An error occurred while processing your request: {e}", "source": "Error"}), 500


@app.route('/summarize', methods=['POST'])
def summarize_answer():
    """Endpoint for summarizing an answer using an SLM, now considering the original question."""
    data = request.get_json()
    original_question = data.get('question', '') 
    text_to_summarize = data.get('text')
    
    if not text_to_summarize:
        return jsonify({"summary": "No text provided for summarization."}), 400

    global slm_model, slm_tokenizer
    if slm_model is None or slm_tokenizer is None:
        return jsonify({"summary": "Summarization model not initialized."}), 500

    try:
        # Craft a prompt that guides the summarization model to answer the original question
        # This makes the summary more relevant to the user's initial query.
        prompt_text = f"Based on the following information, summarize the answer to the question '{original_question}':\n\n{text_to_summarize}"
        
        inputs = slm_tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = slm_model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = slm_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"summary": "An error occurred during summarization."}), 500


# --- Model Initialization on App Startup ---
with app.app_context():
    print("Loading RAG model components...")
    documents_folder = "documents" 
    
    corpus = load_documents(documents_folder)
    
    if corpus:
        chunks = chunk_text(corpus)
        if chunks:
            try:
                # Load the SentenceTransformer model for RAG embeddings
                rag_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("SentenceTransformer loaded successfully!")
                rag_index = create_vector_store(chunks, rag_embedding_model)
                rag_chunks = chunks
                print("RAG model initialized successfully with semantic embeddings!")
            except Exception as e:
                print(f"Error loading SentenceTransformer or creating vector store: {e}")
                print("RAG model will be unavailable.")
        else:
            print("Error: No valid sentences were found after chunking. RAG model will not function.")
    else:
        print("Error: No documents loaded. The RAG model will not be able to answer questions.")
    
    print("Loading Small Language Model for summarization (google/flan-t5-small)...")
    try:
        slm_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        slm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        print("SLM for summarization initialized successfully!")
    except Exception as e:
        print(f"Error loading SLM model: {e}")
        print("Summarization feature will be unavailable.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
