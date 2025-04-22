import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_profiles():
    with open('sample_profiles.json') as f:
        return json.load(f)

def generate_embeddings(text_list):
    return model.encode(text_list)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, 'profiles.index')

def save_text_and_profiles(profiles, texts):
    with open('profile_texts.json', 'w') as f:
        json.dump(texts, f)
    with open('profiles.json', 'w') as f:
        json.dump(profiles, f)

def load_index():
    return faiss.read_index('profiles.index')

def embed_query(query):
    return model.encode([query])[0]

def search(query_embedding, index, top_k=3):
    query_embedding = np.array([query_embedding])
    scores, indices = index.search(query_embedding, top_k)
    return indices[0], scores[0]
