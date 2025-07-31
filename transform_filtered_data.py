import pandas as pd
import joblib
import os
from data_cleaning import data_for_content_filtering
from content_based_filtering import transform_data, save_transformed_data, train_transformer

# Paths
filtered_data_path = "data/collaborative_filtering_data.csv"
save_path = "data/transformed_hybrid_data.npz"
transformer_path = "data/transformer.joblib"

def main(data_path, save_path):
    # Load and clean the filtered data
    filtered_data = pd.read_csv(data_path)
    filtered_data_cleaned = data_for_content_filtering(filtered_data)

    # Load existing transformer or train a new one
    if os.path.exists(transformer_path):
        transformer = joblib.load(transformer_path)
        print(f"Loaded transformer from {transformer_path}")
    else:
        print("Transformer not found. Training new transformer.")
        transformer = train_transformer(filtered_data_cleaned)

    # Transform data
    transformed_data = transform_data(filtered_data_cleaned, transformer)

    # Save the transformed data
    save_transformed_data(save_path,transformed_data)

if __name__ == "__main__":
    main(filtered_data_path, save_path)
    print(f"Transformed hybrid data saved to {save_path}")
