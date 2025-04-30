"""
Final recommendation system for Spotify Million Playlist Dataset Challenge.
"""
import json
import numpy as np
from collections import defaultdict
import re
import string
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os


class HybridRecommender:

    def __init__(self):
        # Collaborative filtering components
        self.track_co_occurrences = defaultdict(lambda: defaultdict(int))
        self.track_counts = defaultdict(int)
        self.track_metadata = {}

        # Content-based components
        self.playlist_name_to_tracks = defaultdict(list)
        self.playlist_names = []

        # Sentence-BERT components
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.playlist_name_embeddings = None

        self.total_playlists = 0

        # Tunable parameters
        self.popularity_weight = 0.0
        self.name_weight = 1.0

    def fit(self, playlists):
        print("Training hybrid recommendation model...")
        self.total_playlists = len(playlists)

        print("Building collaborative filtering model...")
        for playlist in tqdm(playlists):
            track_uris = [track['track_uri'] for track in playlist['tracks']]

            for track in playlist['tracks']:
                track_uri = track['track_uri']
                if track_uri not in self.track_metadata:
                    self.track_metadata[track_uri] = {
                        'name': track['track_name'],
                        'artist': track['artist_name'],
                        'album': track['album_name']
                    }

            for track_uri in track_uris:
                self.track_counts[track_uri] += 1

            for i, track_uri1 in enumerate(track_uris):
                for track_uri2 in track_uris[i+1:]:
                    self.track_co_occurrences[track_uri1][track_uri2] += 1
                    self.track_co_occurrences[track_uri2][track_uri1] += 1

        print("Building content-based model from playlist names...")
        clean_names = []

        for playlist in tqdm(playlists):
            if 'name' in playlist and playlist['name'].strip():
                name = self._clean_playlist_name(playlist['name'])
                clean_names.append(name)
                for track in playlist['tracks']:
                    self.playlist_name_to_tracks[name].append(track['track_uri'])
            else:
                clean_names.append("")

        if any(clean_names):
            self.playlist_names = clean_names
            self.non_empty_playlist_names = [name for name in clean_names if name]
            if self.non_empty_playlist_names:
                self.playlist_name_embeddings = self.sbert_model.encode(self.non_empty_playlist_names,
                                                                        show_progress_bar=True)

        print("Model training complete!")

    def _clean_playlist_name(self, name):
        # cleaning and normalizing the playlist name
        name = name.lower()
        name = re.sub(f'[{string.punctuation}]', ' ', name)
        stopwords = set([
            'playlist', 'songs', 'hits', 'best', 'mix', 'music', 'top', 'new', 'latest', 'greatest',
            'collection', 'compilation', 'the', 'a', 'and'
        ])
        words = name.split()
        words = [word for word in words if word not in stopwords]
        name = ' '.join(words)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def recommend(self, seed_tracks=None, playlist_name=None, n=500, exclude_seeds=True):
        # generating recommendations based on seed tracks and/or playlist name
        scores = defaultdict(float)

        # 1. Collaborative filtering based on seed tracks
        if seed_tracks:
            for seed_track in seed_tracks:
                if seed_track in self.track_co_occurrences:
                    related_tracks = self.track_co_occurrences[seed_track]
                    for track_uri, count in related_tracks.items():
                        scores[track_uri] += count

        # 2. Content-based filtering using playlist name
        if playlist_name and self.playlist_name_embeddings is not None:
            clean_name = self._clean_playlist_name(playlist_name)
            playlist_emb = self.sbert_model.encode([clean_name])
            similarities = cosine_similarity(playlist_emb, self.playlist_name_embeddings).flatten()

            for i, name in enumerate(self.non_empty_playlist_names):
                if name:
                    similarity = similarities[i]
                    if similarity > 0:
                        for track_uri in self.playlist_name_to_tracks[name]:
                            scores[track_uri] += similarity * self.name_weight

        # 3. Popularity fallback if needed
        if not scores and not seed_tracks:
            top_tracks = sorted(self.track_counts.items(), key=lambda x: x[1], reverse=True)
            return [uri for uri, _ in top_tracks[:n]]

        # 4. Final scoring adjustment
        for track_uri in list(scores.keys()):
            scores[track_uri] += self.track_counts.get(track_uri, 0) * self.popularity_weight

        # 5. Sort and filter
        ranked_tracks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if exclude_seeds and seed_tracks:
            ranked_tracks = [(uri, score) for uri, score in ranked_tracks if uri not in seed_tracks]

        return [uri for uri, _ in ranked_tracks[:n]]

    def recommend_for_challenge(self, challenge_playlist, n=500):
        seed_tracks = [track['track_uri'] for track in challenge_playlist['tracks']]
        playlist_name = challenge_playlist.get('name', None)

        recommendations = self.recommend(
            seed_tracks=seed_tracks,
            playlist_name=playlist_name,
            n=n+len(seed_tracks),
            exclude_seeds=True
        )

        return recommendations[:n]


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_submission(challenge_data, model, output_file):
    print(f"Generating recommendations for {len(challenge_data['playlists'])} playlists...")

    with open(output_file, 'w') as f:
        f.write("team_info,Tonalytics,your.email@example.com\n")
        for playlist in tqdm(challenge_data['playlists']):
            pid = playlist['pid']
            recommendations = model.recommend_for_challenge(playlist, n=500)
            f.write(f"{pid},{','.join(recommendations)}\n")

    print(f"Submission file created: {output_file}")


if __name__ == "__main__":
    challenge_file = "../data/challenge_set.json"
    submission_file = "../submissions/final_submission.csv"

    os.makedirs("../submissions", exist_ok=True)

    data = load_data(challenge_file)

    model = HybridRecommender()
    model.fit(data['playlists'])

    model.popularity_weight = 0.0
    model.name_weight = 1.0

    create_submission(data, model, submission_file)

    print("Done! Verify your submission with:")
    print(f"python verify_submission.py {challenge_file} {submission_file}")
