"""
Final recommendation system for Spotify Million Playlist Dataset Challenge.
"""
import json
import numpy as np
from collections import defaultdict
import re
import string
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


class HybridRecommender:
    """
    A hybrid music recommendation system that combines:
    - Collaborative filtering based on track co-occurrences
    - Content-based filtering using playlist names
    """

    def __init__(self):
        # Collaborative filtering components
        self.track_co_occurrences = defaultdict(lambda: defaultdict(int))
        self.track_counts = defaultdict(int)
        self.track_metadata = {}

        # Content-based components
        self.playlist_name_to_tracks = defaultdict(list)
        self.tfidf_matrix = None
        self.playlist_names = []

        # General
        self.total_playlists = 0

        # Tunable parameters
        self.popularity_weight = 0.1
        self.name_weight = 5.0

    def fit(self, playlists):
        """
        Train the model using the provided playlists.

        Args:
            playlists: List of playlists, each containing tracks and optional name
        """
        print("Training hybrid recommendation model...")
        self.total_playlists = len(playlists)

        # Collect track information and co-occurrences (collaborative filtering)
        print("Building collaborative filtering model...")
        for playlist in tqdm(playlists):
            # Get all track URIs in this playlist
            track_uris = [track['track_uri'] for track in playlist['tracks']]

            # Store track metadata
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                if track_uri not in self.track_metadata:
                    self.track_metadata[track_uri] = {
                        'name': track['track_name'],
                        'artist': track['artist_name'],
                        'album': track['album_name']
                    }

            # Count individual tracks
            for track_uri in track_uris:
                self.track_counts[track_uri] += 1

            # Count co-occurrences (pairs of tracks)
            for i, track_uri1 in enumerate(track_uris):
                for track_uri2 in track_uris[i+1:]:
                    self.track_co_occurrences[track_uri1][track_uri2] += 1
                    self.track_co_occurrences[track_uri2][track_uri1] += 1

        # Build content-based model using playlist names
        print("Building content-based model from playlist names...")
        # Extract playlist names and associated tracks
        clean_names = []

        for playlist in tqdm(playlists):
            if 'name' in playlist and playlist['name'].strip():
                name = self._clean_playlist_name(playlist['name'])
                clean_names.append(name)

                # Store tracks for this playlist name
                for track in playlist['tracks']:
                    self.playlist_name_to_tracks[name].append(track['track_uri'])
            else:
                clean_names.append("")  # Empty name for playlists without names

        # Create TF-IDF matrix from playlist names
        if any(clean_names):  # Only if we have non-empty names
            self.playlist_names = clean_names
            vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
            # Filter out empty names
            non_empty_names = [name for name in clean_names if name]
            if non_empty_names:
                self.tfidf_matrix = vectorizer.fit_transform(non_empty_names)
                self.tfidf_feature_names = vectorizer.get_feature_names_out()
                # Create a mapping from clean_names indices to tfidf_matrix indices
                self.name_to_tfidf_idx = {name: i for i, name in enumerate(non_empty_names)}

        print("Model training complete!")

    def _clean_playlist_name(self, name):
        """Clean and normalize a playlist name."""
        # Convert to lowercase
        name = name.lower()
        # Remove punctuation
        name = re.sub(f'[{string.punctuation}]', ' ', name)
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def recommend(self, seed_tracks=None, playlist_name=None, n=500, exclude_seeds=True):
        """
        Generate recommendations based on seed tracks and/or playlist name.

        Args:
            seed_tracks: list of track URIs to base recommendations on
            playlist_name: name of the playlist to base recommendations on
            n: number of recommendations to return
            exclude_seeds: whether to exclude seed tracks from recommendations

        Returns:
            List of recommended track URIs
        """
        # Initialize scores for all tracks
        scores = defaultdict(float)

        # 1. Collaborative filtering based on seed tracks
        if seed_tracks:
            for seed_track in seed_tracks:
                if seed_track in self.track_co_occurrences:
                    # Get co-occurrence counts for this seed track
                    related_tracks = self.track_co_occurrences[seed_track]

                    # Add to scores, weighted by co-occurrence count
                    for track_uri, count in related_tracks.items():
                        # Weight by co-occurrence frequency
                        scores[track_uri] += count

        # 2. Content-based filtering using playlist name
        if playlist_name and hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
            clean_name = self._clean_playlist_name(playlist_name)

            # Find similar playlist names
            if hasattr(self, 'tfidf_feature_names'):
                vectorizer = TfidfVectorizer(vocabulary=self.tfidf_feature_names)
                name_vector = vectorizer.fit_transform([clean_name])

                # Calculate similarity with existing playlist names
                if self.tfidf_matrix.shape[0] > 0:  # Check if we have data
                    similarities = cosine_similarity(name_vector, self.tfidf_matrix).flatten()

                    # Get tracks from similar playlists
                    if hasattr(self, 'name_to_tfidf_idx'):
                        for i, name in enumerate(self.playlist_names):
                            if name and name in self.name_to_tfidf_idx:
                                tfidf_idx = self.name_to_tfidf_idx[name]
                                similarity = similarities[tfidf_idx]
                                if similarity > 0:
                                    for track_uri in self.playlist_name_to_tracks[name]:
                                        # Add score weighted by name similarity
                                        scores[track_uri] += similarity * self.name_weight

        # If no recommendations from collaborative or content-based, use popularity
        if not scores and not seed_tracks:
            # Return most popular tracks
            top_tracks = sorted(self.track_counts.items(),
                               key=lambda x: x[1],
                               reverse=True)
            return [uri for uri, _ in top_tracks[:n]]

        # Final scoring - add popularity as a factor for all tracks
        for track_uri in list(scores.keys()):
            # Add a small popularity boost
            scores[track_uri] += self.track_counts.get(track_uri, 0) * self.popularity_weight

        # Sort tracks by score
        ranked_tracks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out seed tracks if requested
        if exclude_seeds and seed_tracks:
            ranked_tracks = [(uri, score) for uri, score in ranked_tracks
                            if uri not in seed_tracks]

        # Return top n track URIs
        return [uri for uri, _ in ranked_tracks[:n]]

    def recommend_for_challenge(self, challenge_playlist, n=500):
        """
        Generate recommendations for a challenge playlist.

        Args:
            challenge_playlist: a playlist from the challenge set
            n: number of recommendations to return

        Returns:
            List of recommended track URIs, excluding any tracks already in the playlist
        """
        # Extract seed track URIs
        seed_tracks = [track['track_uri'] for track in challenge_playlist['tracks']]

        # Get playlist name if available
        playlist_name = challenge_playlist.get('name', None)

        # Generate recommendations
        recommendations = self.recommend(
            seed_tracks=seed_tracks,
            playlist_name=playlist_name,
            n=n+len(seed_tracks),
            exclude_seeds=True
        )

        # Make sure we return exactly n recommendations
        return recommendations[:n]


def load_data(file_path):
    """Load the challenge set JSON file"""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_submission(challenge_data, model, output_file):
    """Create a submission file for the challenge."""
    print(f"Generating recommendations for {len(challenge_data['playlists'])} playlists...")

    with open(output_file, 'w') as f:
        # Write header with team info
        f.write("team_info,Tonalytics,your.email@example.com\n")

        # Generate recommendations for each playlist
        for playlist in tqdm(challenge_data['playlists']):
            pid = playlist['pid']
            recommendations = model.recommend_for_challenge(playlist, n=500)

            # Write this playlist's recommendations
            f.write(f"{pid},{','.join(recommendations)}\n")

    print(f"Submission file created: {output_file}")


if __name__ == "__main__":
    # Paths
    challenge_file = "../data/challenge_set.json"
    submission_file = "../submissions/final_submission.csv"

    # Create submissions directory if it doesn't exist
    os.makedirs("../submissions", exist_ok=True)

    # Load data
    data = load_data(challenge_file)

    # Train the final model with best parameters
    model = HybridRecommender()
    model.fit(data['playlists'])

    # Set best parameters found during tuning
    model.popularity_weight = 0.1  # Use your best value from tuning
    model.name_weight = 3 # Use your best value from tuning

    # Generate submission
    create_submission(data, model, submission_file)

    print("Done! Verify your submission with:")
    print(f"python verify_submission.py {challenge_file} {submission_file}")