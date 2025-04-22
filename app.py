import streamlit as st
import json
from utils import (
    load_profiles, embed_query,
    load_index, search
)

# Load data
profiles = load_profiles()
index = load_index()

st.title("🤖 AIFindr - People Discovery Engine")

user_query = st.text_input("Describe who you're looking for 👇", 
                           placeholder="Find me a coder who loves anime and talks like Tarantino")

if user_query:
    query_vector = embed_query(user_query)
    match_indices, scores = search(query_vector, index, top_k=3)

    st.subheader("🔍 Top Matches")
    for i, idx in enumerate(match_indices):
        profile = profiles[idx]
        st.markdown(f"""
        ### {profile['name']}
        - **Bio**: {profile['bio']}
        - **Interests**: {', '.join(profile['interests'])}
        - **Vibe**: {profile['vibe']}
        - **Match Score**: {scores[i]:.2f}
        """)
