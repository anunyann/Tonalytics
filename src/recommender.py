import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model and data...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('../models/track_embeddings.npy')
df = pd.read_csv('../models/track_metadata.csv')

def recommend_similar_songs(query_text, model, embeddings, metadata_df, top_k=5):
    query_embedding = model.encode([query_text])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k*5:][::-1]

    top_results = metadata_df.iloc[top_indices][['track_name', 'artist_name', 'playlist_name']].copy()
    top_results['score'] = similarities[top_indices]

    top_results = top_results.drop_duplicates(subset=['track_name', 'artist_name'])

    return top_results.head(top_k)

print("Добредојдовте во Tonalytics!")
while True:
    user_input = input("\n Опишете што сакате да слушате (или напишете exit за да изгасите)\n> ")
    if user_input.lower() == 'exit':
        print("Пријатно!")
        break

    print("\nТоп 5 песни:\n")
    results = recommend_similar_songs(user_input, model, embeddings, df, top_k=5)
    for idx, row in results.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']} [Playlist: {row['playlist_name']}]  (score: {row['score']:.2f})")
