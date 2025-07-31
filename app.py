import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from hybrid_recommendation import HybridRecommendarSystem as hrs 
from content_based_filtering import recommend
from collaborative_filtering_data import collabrative_recommendation

# --- Load Data ---
@st.cache_resource
def load_all_data():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    transformed_data = load_npz("data/transformed_hybrid_data.npz")
    track_ids = np.load("data/track_ids.npy", allow_pickle=True)
    filtered_data = pd.read_csv("data/collaborative_filtering_data.csv")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    return songs_data, transformed_data, track_ids, filtered_data, interaction_matrix

# Load everything
songs_data, transformed_data, track_ids, filtered_data, interaction_matrix = load_all_data()

# Save in session state (optional for reuse)
st.session_state.songs_data = songs_data
st.session_state.transformed_data = transformed_data
st.session_state.track_ids = track_ids
st.session_state.filtered_data = filtered_data
st.session_state.interaction_matrix = interaction_matrix

# --- Streamlit UI ---
st.title("🎵 Song Recommendation System")

song_name_input = st.text_input("Enter a song name:", "").strip()
artist_name_input = st.text_input("Enter the artist name:", "").strip()
k_recommendations = st.slider("Number of recommendations", 5, 20, 10)

# Lowercased versions for comparison
song_name = song_name_input.lower()
artist_name = artist_name_input.lower()

# Check if song exists in collaborative dataset
song_exists = (
    (filtered_data['name'].str.lower() == song_name) &
    (filtered_data['artist'].str.lower() == artist_name)
).any()

# Show filtering options based on song presence
if song_exists:
    filtering_type = st.selectbox("Select filtering type:",
                                  ["Content-Based", "Collaborative", "Hybrid Recommender System"])
    diversity = st.slider("Diversity in Recommendation (0=More Collaborative, 10=More Content)", 0, 10, 5)
    content_based_weight = diversity / 10
    collaborative_weight = 1 - content_based_weight
else:
    st.info("Song not found in collaborative data. Defaulting to Content-Based Filtering.")
    filtering_type = "Content-Based"

# --- Display Recommendations ---
def display_recommendations(recommendations_df: pd.DataFrame):
    if recommendations_df is not None and not recommendations_df.empty:
        st.subheader("Recommended Songs:")
        for _, rec in recommendations_df.iterrows():
            st.markdown(f"**🎧 {rec['name'].title()}** by *{rec['artist'].title()}*")
            if pd.notna(rec.get("spotify_preview_url")) and rec["spotify_preview_url"]:
                st.audio(rec["spotify_preview_url"], format="audio/mp3")
            st.write("---")
    else:
        st.error(f"❌ No recommendations found for '{song_name_input}' by '{artist_name_input}'.")

# --- Recommendation Logic ---
if filtering_type == "Content-Based":
    if st.button("Get Content-Based Recommendations"):
        if not song_name_input:
            st.warning("Please enter a song name.")
        else:
            recommendations = recommend(song_name_input, songs_data, transformed_data, k_recommendations)
            display_recommendations(recommendations)

elif filtering_type == "Collaborative":
    if st.button("Get Collaborative Recommendations"):
        if not song_name_input or not artist_name_input:
            st.warning("Please enter both song name and artist name.")
        else:
            recommendations = collabrative_recommendation(
                song_name=song_name_input,
                artist_name=artist_name_input,
                track_ids=track_ids,
                songs_data=filtered_data,
                interaction_matrix=interaction_matrix,
                k=k_recommendations
            )
            display_recommendations(recommendations)

elif filtering_type == "Hybrid Recommender System":
    if st.button("Get Hybrid Recommendations"):
        if not song_name_input or not artist_name_input:
            st.warning("Please enter both song name and artist name.")
        else:
            # ✅ FIXED: Only pass init arguments here
            recommender = hrs(
                number_of_recommendations=k_recommendations,
                weight_content=content_based_weight
            )

            # ✅ All other arguments passed to .recommendation()
            recommendations = recommender.recommendation(
                song_name=song_name_input,
                artist_name=artist_name_input,
                track_ids=track_ids,
                transformed_matrix=transformed_data,
                songs_data=filtered_data,
                interaction_matrix=interaction_matrix
            )
            display_recommendations(recommendations)
