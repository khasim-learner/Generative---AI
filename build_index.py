from utils import load_profiles, generate_embeddings, build_faiss_index, save_text_and_profiles

# Load sample profiles
profiles = load_profiles()

# Combine bio + vibe to form searchable text
texts = [p['bio'] + " " + p['vibe'] for p in profiles]

# Generate embeddings using SentenceTransformer
embeddings = generate_embeddings(texts)

# Save vector DB to 'profiles.index'
build_faiss_index(embeddings)

# Save backup of texts and metadata
save_text_and_profiles(profiles, texts)

print("✅ FAISS index created and saved as 'profiles.index'")
