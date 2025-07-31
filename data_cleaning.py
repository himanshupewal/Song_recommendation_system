import pandas as pd

DATA_PATH = "data/Music Info.csv"
OUTPUT_PATH = "data/cleaned_data.csv"

def clean_data(data):
    return (
        data.drop_duplicates(subset="spotify_id") # This is the fixed line: 'errors="ignore"' removed
            .drop(columns=["genre", "spotify_id"], errors="ignore")
            .fillna({"tags": "no_tags"})
            .assign(
                name=lambda x: x["name"].str.lower(),
                artist=lambda x: x["artist"].str.lower(),
                tags=lambda x: x["tags"].str.lower(),
            )
            .reset_index(drop=True)
    )

def data_for_content_filtering(data):
    return(
        data.drop(columns = ["track_id","name","spotify_preview_url"])
    )
def main():
    data = pd.read_csv(DATA_PATH)
    cleaned = clean_data(data)
    cleaned.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()