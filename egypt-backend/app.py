import os
import pickle
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from groq._base_client import DefaultHttpxClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",                     
    "https://egyptia-two.vercel.app/" 
])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY, http_client=DefaultHttpxClient())

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBEDDING_URL = (
    "https://api-inference.huggingface.co/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(PROJECT_ROOT, "rag_data.index")
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "rag_chunks.pkl")

rag_index = None
rag_chunks = None

def load_rag_assets():
    """Load pre-computed FAISS index and text chunks from disk."""
    global rag_index, rag_chunks

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Required assets (rag_data.index / rag_chunks.pkl) not found. "
            "Please run scripts/generate_assets.py locally first."
        )

    rag_index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        rag_chunks = pickle.load(f)

    print(f"Loaded FAISS index ({rag_index.ntotal} vectors) and {len(rag_chunks)} chunks.")

# Initialize assets on startup.
load_rag_assets()

def get_query_embedding(query, timeout=15):
    """Fetch query embedding from Hugging Face Inference API."""
    if not HF_API_TOKEN:
        print("HF_API_TOKEN is missing, skipping retrieval.")
        return None

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": query, "options": {"wait_for_model": True}}

    try:
        response = requests.post(
            HF_EMBEDDING_URL, headers=headers, json=payload, timeout=timeout
        )
    except requests.exceptions.RequestException as e:
        print(f"HF Inference API request failed: {e}")
        return None

    if response.status_code != 200:
        print(f"HF Inference API error {response.status_code}: {response.text[:300]}")
        return None

    try:
        output = response.json()
    except ValueError:
        print("Failed to parse HF Inference API response as JSON.")
        return None

    embedding = _normalize_hf_embedding_output(output)
    if embedding is None:
        print(f"Unexpected embedding format: {type(output)}")
        return None

    return embedding.reshape(1, -1).astype("float32")

def _normalize_hf_embedding_output(output):
    """Normalize HF output variants into a flat sentence embedding vector."""
    try:
        arr = np.array(output, dtype="float32")
    except (ValueError, TypeError):
        return None

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.mean(axis=0)

    return None

def retrieve_chunks(query_embedding, index, chunks, k=5, relevance_threshold=0.8):
    """Search FAISS index and filter results by relevance threshold."""
    if query_embedding is None:
        return [], np.array([])

    distances, indices = index.search(query_embedding, k)

    if indices.size == 0:
        return [], np.array([])

    filtered_chunks_text = []
    filtered_distances = []

    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        distance = distances[0][i]

        if chunk_index == -1:
            continue

        if distance < relevance_threshold:
            chunk_text = chunks[chunk_index]
            if chunk_text.strip():
                filtered_chunks_text.append(chunk_text)
                filtered_distances.append(distance)
        else:
            break

    return filtered_chunks_text, np.array(filtered_distances)

def call_groq_llm(query, context=None):
    """Generate response using Groq LLM with contextual guidance."""
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
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        return f"Error: {e}"

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        if not data:
            print("Empty request body received.")
            return jsonify({"answer": "No data"}), 400

        query = data.get("query")
        if not query:
            print("Missing query parameter.")
            return jsonify({"answer": "No query provided"}), 400

        print(f"Processing query: {query}")
        query_embedding = get_query_embedding(query)

        print("Searching vector index...")
        retrieved_chunks, distances = retrieve_chunks(
            query_embedding, rag_index, rag_chunks
        )

        context = " ".join(retrieved_chunks) if retrieved_chunks else None

        print("Generating response via Groq...")
        answer_text = call_groq_llm(query, context)

        return jsonify({"answer": answer_text, "source": "RAG System"})

    except Exception as e:
        print(f"Unhandled error in /ask: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize_answer():
    data = request.get_json()
    original_question = data.get("question", "")
    text_to_summarize = data.get("text", "")

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
                {"role": "user", "content": f"Text to refine: {text_to_summarize}"},
            ],
            temperature=0.5,
        )

        refined_text = completion.choices[0].message.content.strip()
        return jsonify({"summary": refined_text})

    except Exception as e:
        print(f"Summarize Error: {e}")
        return jsonify({"summary": f"Error: {e}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "index_vectors": rag_index.ntotal if rag_index else 0,
            "chunks_loaded": len(rag_chunks) if rag_chunks else 0,
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)