import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from custom_transformer import CountEncoder
from scipy.sparse import save_npz, csr_matrix
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# Paths
CLEANED_PATH = "data/cleaned_data.csv"
TRANSFORMER_PATH = "data/transformer.joblib"
TRANSFORMED_DATA_PATH = "data/transformed_data.npz"

# tag_utils.py
def extract_tags_column(df):
    return df['tags']


# Column Groups
frequency_encode_cols = ["year"]
ohe_cols = ["artist", "time_signature", "key"]
tfidf_col = "tags"
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "valence"]

def train_transformer(data):
    """Trains and returns a ColumnTransformer."""
    transformer = ColumnTransformer(
        transformers=[
            ('freq', CountEncoder(), frequency_encode_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols),
            ('tfidf', Pipeline([
                ('extract', FunctionTransformer(extract_tags_column, validate=False)),
                ('tfidf', TfidfVectorizer())
            ]), ['tags']),
            ('std', StandardScaler(), standard_scale_cols),
            ('minmax', MinMaxScaler(), min_max_scale_cols)
        ],
        remainder='drop'
    )

    transformer.fit(data)
    joblib.dump(transformer, TRANSFORMER_PATH)
    return transformer  # ✅ FIXED: return the trained transformer

def transform_data(data: pd.DataFrame, transformer):
    """Applies the transformer and returns the transformed data as a sparse matrix."""
    if transformer is None:
        try:
            transformer = joblib.load(TRANSFORMER_PATH)
        except FileNotFoundError:
            print(f"Error: Transformer not found at {TRANSFORMER_PATH}. Please train it first.")
            return None

    transformed_data = transformer.transform(data)

    # ✅ Ensure it's a sparse matrix
    if not isinstance(transformed_data, csr_matrix):
        transformed_data = csr_matrix(transformed_data)

    return transformed_data

def save_transformed_data(transformed_data, save_path):
    """Saves transformed sparse data to disk."""
    save_npz(transformed_data,save_path)


def calculate_similarity(input_vector, data):
    """Returns cosine similarity scores."""
    if input_vector.ndim == 1:
        input_vector = input_vector.reshape(1, -1)
    return cosine_similarity(input_vector, data)

def recommend(song_name, songs_data, transformed_data_features, k=10):
    """Recommends songs based on content-based filtering."""
    song_name = song_name.lower()
    song_row = songs_data.loc[songs_data['name'] == song_name]

    if song_row.empty:
        print(f"Song '{song_name}' not found.")
        return pd.DataFrame(columns=["name", "artist", "spotify_preview_url"])

    song_index = song_row.index[0]
    input_vector = transformed_data_features[song_index].reshape(1, -1)
    similarity_scores = calculate_similarity(input_vector, transformed_data_features)
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:-1][::-1]
    top_k_songs = songs_data.iloc[top_k_songs_indexes]
    return top_k_songs[["name", "artist", "spotify_preview_url"]].reset_index(drop=True)

# === DVC Stage Execution ===
if __name__ == "__main__":
    print("Starting content-based filtering data transformation stage...")
    
    df = pd.read_csv(CLEANED_PATH)
    print(f"Loaded data from {CLEANED_PATH} with {len(df)} rows.")

    # Load or train transformer
    transformer = None
    if os.path.exists(TRANSFORMER_PATH):
        try:
            transformer = joblib.load(TRANSFORMER_PATH)
            print(f"Loaded existing transformer from {TRANSFORMER_PATH}")
        except Exception as e:
            print(f"Error loading transformer: {e}. Retraining...")
            transformer = train_transformer(df)
    else:
        transformer = train_transformer(df)

    if transformer is None:
        print("Failed to train or load transformer. Exiting.")
        exit(1)

    # Transform and save the data
    transformed_features = transform_data(df, transformer)
    if transformed_features is None:
        print("Failed to transform data. Exiting.")
        exit(1)

    save_transformed_data(TRANSFORMED_DATA_PATH,transformed_features)
    print(f"Transformed data saved to {TRANSFORMED_DATA_PATH}")
    print("Content-based filtering data transformation stage completed.")
