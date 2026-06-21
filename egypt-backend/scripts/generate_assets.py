import os
import re
import sys
import pickle
import numpy as np
import faiss
import nltk
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_FOLDER = os.path.join(PROJECT_ROOT, "documents")
INDEX_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "rag_data.index")
CHUNKS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "rag_chunks.pkl")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(folder_path):
    """Load text content from all .txt files in the target directory."""
    documents = []
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return ""

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    if not documents:
        print(f"Warning: No valid .txt files found in '{folder_path}'")
        return ""

    return " ".join(documents)


def chunk_text(text):
    """Split text corpus into sentences using regular expressions."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def create_vector_store(chunks, embedding_model):
    """Encode text chunks and initialize FAISS index structures."""
    chunk_embeddings = embedding_model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=True
    ).astype("float32")

    d = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(chunk_embeddings)

    return index


def save_assets(index, chunks):
    """Persist the generated vector index and chunk mappings to disk."""
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print(f"Saved FAISS index to {INDEX_OUTPUT_PATH}")

    with open(CHUNKS_OUTPUT_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunk mappings to {CHUNKS_OUTPUT_PATH}")


def main():
    nltk.download("punkt", quiet=True)

    print("Loading source documents...")
    corpus = load_documents(DOCUMENTS_FOLDER)
    if not corpus:
        print("Process aborted: Empty document corpus.")
        sys.exit(1)

    print("Segmenting text into database chunks...")
    chunks = chunk_text(corpus)
    print(f"Generated {len(chunks)} chunks.")
    if not chunks:
        print("Process aborted: Zero text chunks generated.")
        sys.exit(1)

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding database chunks and building FAISS index...")
    index = create_vector_store(chunks, embedding_model)
    print(f"Index created with {index.ntotal} vectors (dimension: {index.d}).")

    print("Writing artifact payloads to project root...")
    save_assets(index, chunks)
    print("Asset generation complete.")


if __name__ == "__main__":
    main()