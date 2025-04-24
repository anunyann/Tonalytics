import json
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_challenge_data(file_path):
    """Load the challenge set JSON file"""
    print(f"Loading challenge data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Challenge set loaded. Version: {data['version']}")
    print(f"Number of playlists: {len(data['playlists'])}")
    return data


def analyze_challenge_playlists(data):
    """Analyze basic statistics about the challenge playlists"""
    playlists = data['playlists']

    # Count challenge categories
    categories = Counter()
    for playlist in playlists:
        # Categorize based on num_samples and whether it has a name
        has_name = 'name' in playlist
        num_samples = playlist['num_samples']

        if num_samples == 0 and has_name:
            category = "Title only"
        elif num_samples == 1 and has_name:
            category = "Title + first track"
        elif num_samples == 5 and has_name:
            category = "Title + first 5 tracks"
        elif num_samples == 5 and not has_name:
            category = "First 5 tracks (no title)"
        elif num_samples == 10 and has_name:
            category = "Title + first 10 tracks"
        elif num_samples == 10 and not has_name:
            category = "First 10 tracks (no title)"
        elif num_samples == 25 and has_name and playlist['tracks'][0]['pos'] == 0:
            category = "Title + first 25 tracks"
        elif num_samples == 25 and has_name:
            category = "Title + 25 random tracks"
        elif num_samples == 100 and has_name and playlist['tracks'][0]['pos'] == 0:
            category = "Title + first 100 tracks"
        elif num_samples == 100 and has_name:
            category = "Title + 100 random tracks"
        else:
            category = "Other"

        categories[category] += 1

    print("\nChallenge Categories:")
    for category, count in categories.most_common():
        print(f"  {category}: {count} playlists")

    return categories


def collect_track_statistics(data):
    """Collect statistics about tracks in the challenge set"""
    all_tracks = {}
    artist_counts = Counter()
    album_counts = Counter()

    # Collect all tracks and count occurrences
    for playlist in data['playlists']:
        for track in playlist['tracks']:
            track_uri = track['track_uri']
            artist_uri = track['artist_uri']
            album_uri = track['album_uri']

            if track_uri not in all_tracks:
                all_tracks[track_uri] = {
                    'name': track['track_name'],
                    'artist': track['artist_name'],
                    'album': track['album_name'],
                    'count': 0
                }

            all_tracks[track_uri]['count'] += 1
            artist_counts[artist_uri] += 1
            album_counts[album_uri] += 1

    print(f"\nUnique tracks: {len(all_tracks)}")
    print(f"Unique artists: {len(artist_counts)}")
    print(f"Unique albums: {len(album_counts)}")

    # Find most common tracks
    top_tracks = sorted(all_tracks.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    print("\nTop 10 most common tracks:")
    for uri, track_info in top_tracks:
        print(f"  {track_info['name']} by {track_info['artist']} - {track_info['count']} occurrences")

    # Find most common artists
    top_artists = artist_counts.most_common(10)
    print("\nTop 10 most common artists:")
    for uri, count in top_artists:
        # Find a track with this artist to get the name
        for track_data in all_tracks.values():
            if track_data['artist'] and track_data['artist'] != "":
                print(f"  {track_data['artist']} - {count} occurrences")
                break

    return all_tracks, artist_counts, album_counts


if __name__ == "__main__":
    # Update this path to where your challenge_set.json file is located
    challenge_file = "../data/challenge_set.json"

    # Check if file exists
    if not os.path.exists(challenge_file):
        print(f"Error: File not found: {challenge_file}")
        print("Please download the challenge_set.json file and place it in the data directory.")
        exit(1)

    # Load and analyze the data
    challenge_data = load_challenge_data(challenge_file)
    categories = analyze_challenge_playlists(challenge_data)
    tracks, artists, albums = collect_track_statistics(challenge_data)

    print("\nData analysis complete!")