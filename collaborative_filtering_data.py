import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy.sparse import csr_matrix,save_npz
from sklearn.metrics.pairwise import cosine_similarity


track_ids_path = 'data/track_ids.npy'
filtered_data_save_path = 'data/collaborative_filtering_data.csv'
interaction_matrix_save_path = 'data/interaction_matrix.npz'


song_data_path = r"C:\Users\A\Desktop\song_recomand_system\data\cleaned_data.csv"
user_listening_data_path = r"C:\Users\A\Desktop\song_recomand_system\data\User Listening History.csv"

def filter_songs_data(songs_data:pd.DataFrame,track_ids:list,save_df_path:str) -> pd.DataFrame:
    """
    Filter songs data based on track IDs and save the filtered data.
    """
    filtered_data = songs_data[songs_data['track_id'].isin(track_ids)]
    filtered_data.reset_index(drop=True, inplace=True)
    save_pandas_data_to_csv(filtered_data, save_df_path)
    
    return filtered_data


def save_pandas_data_to_csv(data: pd.DataFrame, path: str):
    data.to_csv(path, index=False)



def save_sparse_matrix(matrix: csr_matrix, path: str):
    """
    Save a sparse matrix to a file.
    """
    save_npz(path, matrix)


def create_interaction_matrix(history_data:dd.DataFrame,track_ids_save_path,save_matrix_path:str) -> csr_matrix:

    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)

    df = df.categorize(columns=["user_id","track_id"])

    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes

    track_ids = df['track_id'].cat.categories.values

    np.save(track_ids_save_path,track_ids,allow_pickle=True)


    df = df.assign(user_idx = user_mapping, track_idx = track_mapping)


    interaction_matrix  = df.groupby(["track_idx","user_idx"])['playcount'].sum().reset_index()
    interaction_matrix = interaction_matrix.compute()


    row_indices = interaction_matrix["track_idx"]
    col_indices = interaction_matrix["user_idx"]
    data_values = interaction_matrix["playcount"]

    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()


    interaction_matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=(n_tracks, n_users))

    save_sparse_matrix(interaction_matrix, save_matrix_path)



def collabrative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k=10):

    song_name = song_name.lower()
    artist_name = artist_name.lower()

    song_row = songs_data.loc[(songs_data['name']== song_name) & (songs_data['artist'] == artist_name)]


    input_track_id = song_row['track_id'].values.item()

    ind = np.where(track_ids == input_track_id)[0].item()

    input_array = interaction_matrix[ind]

    similarity_score = cosine_similarity(input_array, interaction_matrix)

    recommendation_indices = np.argsort(similarity_score.ravel())[-k-1:-1][::-1]

    recommendation_track_ids = track_ids[recommendation_indices]

    top_scores =np.sort(similarity_score.ravel())[-k-1:-1][::-1]

    scores_df = pd.DataFrame({"track_id": recommendation_track_ids.tolist(), "score": top_scores})
    top_k_songs=(
        songs_data
        .loc[songs_data["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df,on="track_id")
        .sort_values(by="score",ascending=False)
        .drop(columns=['track_id',"score"])
        .reset_index(drop=True)
    )

    return top_k_songs


def main():

    user_data = dd.read_csv(user_listening_data_path)

    unique_track_ids = user_data['track_id'].unique().compute().tolist()

    songs_data = pd.read_csv(song_data_path)

    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)

    create_interaction_matrix(user_data, track_ids_path, interaction_matrix_save_path)


if __name__ == "__main__":
    main()