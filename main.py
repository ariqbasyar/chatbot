import json
import os
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = "all-mpnet-base-v2" # Model for embedding

# Load embedding model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")
    raise

def get_openai_response(user_message, context):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [
        {
            "role": "user",
            "content": f"Answer based on the following context: {context}\n\nUser message: {user_message}"
        }
    ]
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating response with OpenAI: {e}")
        return "Sorry, I encountered an error processing your request."


def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"Knowledge base file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON in: {file_path}")
        return []


def embed_knowledge_base(knowledge_base):
    logging.debug("Starting embedding process...")
    embeddings = []
    texts = []
    for item in knowledge_base:
         if "content" in item:
             texts.append(item["content"])

    try:
         embeddings = embedding_model.encode(texts)
         embeddings = np.array(embeddings).astype(np.float32)
         logging.debug(f"Embeddings created: {embeddings.shape}")
         return embeddings, texts
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return None, None


def create_faiss_index(embeddings):
    logging.debug("Creating FAISS index...")
    try:
         d = embeddings.shape[1] # embedding dimension
         index = faiss.IndexFlatL2(d)
         index.add(embeddings)
         logging.debug("FAISS index created.")
         return index
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        return None

def find_relevant_context(query, faiss_index, embeddings, texts, top_k=3):
    try:
        query_embedding = embedding_model.encode(query)
        query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)  # Reshape to (1, embedding_dim)
        D, I = faiss_index.search(query_embedding, top_k)
        if len(I) > 0:
            relevant_texts = [texts[i] for i in I[0]]
            logging.debug(f"Relevant texts found: {len(relevant_texts)} - {relevant_texts}")
            return "\n\n".join(relevant_texts)
        else:
          logging.debug("No matching texts found.")
          return None
    except Exception as e:
        logging.error(f"Error in finding relevant context: {e}")
        return None


def main():
    knowledge_base_file = 'texts.json'
    logging.info(f"Loading knowledge base from {knowledge_base_file}...")
    knowledge_base = load_knowledge_base(knowledge_base_file)

    if not knowledge_base:
        logging.error("Knowledge base loading failed. Exiting.")
        return

    logging.info("Embedding and indexing...")
    embeddings, texts = embed_knowledge_base(knowledge_base)
    if embeddings is None:
        return
    faiss_index = create_faiss_index(embeddings)
    if faiss_index is None:
        return

    logging.info("Chat bot initialized. Ready for input.")

    while True:
        user_input = input("Enter your message (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        start_time = time.time()
        # Perform RAG to find relevant context for user input
        relevant_context = find_relevant_context(user_input, faiss_index, embeddings, texts)
        if relevant_context:
            openai_response = get_openai_response(user_input, relevant_context)
        else:
            openai_response = get_openai_response(user_input, "No relevant content was found")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"\n\nOpenAI Response: {openai_response}\n\nTook: {elapsed_time:.2f} seconds\n\n")

if __name__ == "__main__":
    main()