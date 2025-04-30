spotify_data_exploration.ipnyb
purpose: understand and explore the structure and properties of the dataset
I loaded the dataset, analyzed how many playlists had a title vs. no title and how many had 1, 5, 10, 25 or 100 seed tracks. 
Then, I categorized them into buckets (This helped with understanding the variation in playlist coverage, so we can desing a model that can work when only a title is provided or when there are very few tracks).
I counted how many:
- Unique tracks
- Unique artists
- Unique albums
- How often each track appeared in the dataset
- Identified top 10 most common tracks and artists
What I saw:
- Many playlists had just a name and a few tracks
- Some playlists had no name at all
- A few popular tracks appeared hundreds or thousands of times
- Certain artists appeared extremely frequently
Based on this, I decided to build a hybrid recommender system that:
- Uses seed tracks (collaborative filtering)
- Uses playlist names (content-based filtering with NLP)
- Falls back to popularity when needed

data_processing.py  - I later deleted this and started using only inline logic
purpose: helper functions for preparing playlists into training and test sets, including splitting tracks and extracting seed/hidden items
I defined train-test split logic, controlled the minimum playlist length, set the ratio for splitting tracks.

model_development.ipynb
purpose: first implementation and testing of my hybrid recommendation model 
I loaded the json, split it into seed tracks and hidden tracks. The purpose was to simulate "If I know the first few tracks, can I predict the rest?"
I looped through playlists to count how often two tracks appeared together, and I created a co-occurence dictionary which was the base of my collaborative model.
I cleaned the playlist names and encoded them with Sentence-BERT. 
For each playlists, I calculated co-occurence score, then added in tracks from similar playlists (via name similarity), then added track popularity as a weight. 
The final result is top-500 track recommendations per playlist.
I evaluated using Recall@100
Before implementing Sentence-BERT, I was only using Collaborative Filtering via Track Co-occurence and Popularity-Based Scoring. I tried TF-IDF based content filtering but it was weak because many playlist names were short or empty and it couldn't understand the word meaning.

model_tuning.ipynb
purpose: perform hyperparameter tuning to improve the hybrid recommender's performance (specifically Recall@100), by testing combinations of name and popularity weights
I used my recommender to test how much weight to give to playlist name similarity, how much to boost the popular tracks and how much these changes impact Recall@100
I performed a grid search - got nan :)
