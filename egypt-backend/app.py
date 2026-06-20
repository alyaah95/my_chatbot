# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import faiss
import numpy as np
import os
import re

# Import necessary libraries for the Small Language Models

from sentence_transformers import SentenceTransformer 



from groq import Groq

from dotenv import load_dotenv
load_dotenv()


# Initialize Flask app
app = Flask(__name__)
CORS(app) 



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


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



def call_groq_llm(query, context=None):
    try:
        
        system_instructions = (
            "You are a highly knowledgeable and professional AI assistant. "
            "Your goal is to provide direct, accurate, and comprehensive answers. "
            "If the provided context contains relevant information, use it. "
            "If the context is missing information or irrelevant, answer using your own broad general knowledge. "
            "CRITICAL: Do not ever mention phrases like 'according to the text', 'the provided context does not mention', "
            "or 'based on the files'. Just provide the answer directly as if you already know it."
        )

       
        if context:
            user_content = f"Context: {context}\n\nQuestion: {query}"
        else:
            user_content = query

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DEBUG] Groq API Error: {e}")
        return f"Error: {e}"


# --- API Endpoints ---
@app.route('/ask', methods=['POST'])
def ask_question():
    
    print("\n[DEBUG] Connection established! Processing request...")
    try:
        data = request.get_json()
        if not data:
            print("[DEBUG] No data received in request.")
            return jsonify({"answer": "No data"}), 400
            
        query = data.get('query')
        print(f"[DEBUG] User asked: {query}")


        print("[DEBUG] Searching in FAISS index...")
        retrieved_chunks, distances = retrieve_chunks(query, rag_index, rag_embedding_model, rag_chunks)
        
        context = " ".join(retrieved_chunks) if retrieved_chunks else None
        

        print("[DEBUG] Calling Groq API...")
        answer_text = call_groq_llm(query, context)
        
        print("[DEBUG] Response ready!")
        return jsonify({"answer": answer_text, "source": "RAG System"})

    except Exception as e:
        print(f"[DEBUG] CRITICAL ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_answer():
    data = request.get_json()
    original_question = data.get('question', '') 
    text_to_summarize = data.get('text', '')
    
    if not text_to_summarize:
        return jsonify({"summary": "No text provided."}), 400

    try:

        summary_system_prompt = (
        "You are a factual editor. Your task is to summarize the following text.\n" 
        f"The original question was: '{original_question}'.\n" 
        "Strict Rules:\n" 
        "- Keep only the direct answer.\n"
        "- Make it exactly 2-3 sentences long.\n"
        "- Delete all introductory phrases.\n"
        "Output Rule: Start the answer directly with facts." 
    )

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": f"Text to refine: {text_to_summarize}"}
            ],
            temperature=0.5, 
        )
        
        refined_text = completion.choices[0].message.content.strip()
        return jsonify({"summary": refined_text})

    except Exception as e:
        print(f"[DEBUG] Summarize Error: {e}")
        return jsonify({"summary": f"Error: {e}"}), 500

    

if __name__ == '__main__':
    print("--- STEP 1: Loading Documents ---")
    documents_folder = "../documents"
    corpus = load_documents(documents_folder)
    
    if not corpus:
        print("Error: No documents found in 'documents' folder!")
    else:
        print("--- STEP 2: Chunking Text ---")
        rag_chunks = chunk_text(corpus)
        
        print("--- STEP 3: Loading Embedding Model (Downloading if first time) ---")

        rag_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer loaded successfully!")
        
        print("--- STEP 4: Creating Vector Store (Encoding chunks) ---")

        rag_index = create_vector_store(rag_chunks, rag_embedding_model)
        
        print("--- STEP 5: Starting Flask Server ---")

        app.run(host='0.0.0.0', port=5000, debug=False)