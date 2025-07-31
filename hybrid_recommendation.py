import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommendarSystem:

    def __init__(self, number_of_recommendations: int, weight_content: float):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content = weight_content
        self.weight_collaborative = 1 - weight_content

    def __calculate_content_based_similarity(self, song_name, artist_name, songs_data, transformed_matrix):
        song_row = songs_data.loc[
            (songs_data['name'].str.lower() == song_name.lower()) &
            (songs_data['artist'].str.lower() == artist_name.lower())
        ]
        if song_row.empty:
            return None, None
        
        song_index = song_row.index[0]
        input_vector = transformed_matrix[song_index]
        similarity_scores = cosine_similarity(input_vector, transformed_matrix).ravel()
        content_ids = songs_data["track_id"].values
        return similarity_scores, content_ids

    def __calculate_collaborative_similarity(self, song_name, artist_name, songs_data, track_ids, interaction_matrix):
        song_row = songs_data.loc[
            (songs_data['name'].str.lower() == song_name.lower()) &
            (songs_data['artist'].str.lower() == artist_name.lower())
        ]
        if song_row.empty:
            return None, None

        input_track_id = song_row["track_id"].values[0]
        try:
            ind = np.where(track_ids == input_track_id)[0][0]
        except IndexError:
            return None, None

        input_vector = interaction_matrix[ind]
        similarity_scores = cosine_similarity(input_vector, interaction_matrix).ravel()
        return similarity_scores, track_ids

    def __normalize_similarity(self, scores):
        if scores is None:
            return None
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def __weight_combination(self, content_scores, content_ids, collab_scores, collab_ids):
        content_df = pd.DataFrame({"track_id": content_ids, "content_score": content_scores})
        collab_df = pd.DataFrame({"track_id": collab_ids, "collab_score": collab_scores})
        merged = pd.merge(content_df, collab_df, on="track_id", how="inner")

        merged["weighted_score"] = (
            self.weight_content * merged["content_score"] +
            self.weight_collaborative * merged["collab_score"]
        )

        return merged

    def recommendation(self, song_name, artist_name, track_ids, transformed_matrix, songs_data, interaction_matrix):
        content_based_similarities, content_ids = self.__calculate_content_based_similarity(
            song_name, artist_name, songs_data, transformed_matrix)

        collab_based_similarities, collab_ids = self.__calculate_collaborative_similarity(
            song_name, artist_name, songs_data, track_ids, interaction_matrix)

        if content_based_similarities is None or collab_based_similarities is None:
            return pd.DataFrame()

        content_scores = self.__normalize_similarity(content_based_similarities)
        collab_scores = self.__normalize_similarity(collab_based_similarities)

        combined_scores_df = self.__weight_combination(content_scores, content_ids, collab_scores, collab_ids)

        top_scores_df = combined_scores_df.sort_values(by="weighted_score", ascending=False).head(self.number_of_recommendations)

        top_k_songs = pd.merge(top_scores_df, songs_data, on="track_id", how="left")[[
            "name", "artist", "spotify_preview_url"
        ]].drop_duplicates().reset_index(drop=True)

        return top_k_songs
