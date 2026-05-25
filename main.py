import pandas as pd
import numpy as np
import pickle

from src.preprocessing import select_features, scale_features
from src.clustering import train_kmeans, save_model
from src.predict import predict_song, FEATURE_COLUMNS

def train_pipeline():

    print("Loading spotify_data.csv (1.16M tracks)...")
    df = pd.read_csv("data/raw/spotify_data.csv")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    df = df.dropna(subset=FEATURE_COLUMNS)
    df = df.drop_duplicates(subset="track_id", keep="first")
    print(f"After cleaning: {len(df)} tracks")

    X = select_features(df)
    X_scaled, scaler = scale_features(X)

    print("Training KMeans with k=5...")
    model = train_kmeans(X_scaled, k=5)
    save_model(model, path="models/kmeans_model.pkl")

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    labels = model.labels_
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels

    print("\n--- Cluster Summary ---")
    for c in sorted(df_with_labels["cluster"].unique()):
        subset = df_with_labels[df_with_labels["cluster"] == c]
        print(f"\nCluster {c} ({len(subset)} songs):")
        for feat in FEATURE_COLUMNS:
            print(f"  {feat:20s}: {subset[feat].mean():.3f}")

    print("\n--- Sample songs per cluster ---")
    for c in sorted(df_with_labels["cluster"].unique()):
        subset = df_with_labels[df_with_labels["cluster"] == c]
        samples = subset.sample(min(3, len(subset)), random_state=42)
        print(f"\nCluster {c}:")
        for _, row in samples.iterrows():
            artist = row.get("artist_name", row.get("artists", "Unknown"))
            track = row.get("track_name", "Unknown")
            print(f"  - {artist} — {track}")

    print("\nTraining complete. Model and scaler saved.")

def test_prediction():

    sample_songs = [
        ([0.75, 0.70, -5.5, 0.08, 0.12, 0.0, 0.15, 0.60, 120], "Generic pop"),
        ([0.85, 0.90, -3.0, 0.05, 0.01, 0.0, 0.30, 0.80, 150], "High energy/EDM"),
        ([0.30, 0.20, -15.0, 0.03, 0.90, 0.0, 0.10, 0.25, 80], "Acoustic/Calm"),
        ([0.60, 0.80, -4.0, 0.35, 0.05, 0.0, 0.50, 0.50, 130], "Hip-hop/Rap"),
    ]

    for features, desc in sample_songs:
        result = predict_song(features)
        print(f"{desc:20s} → {result}")

if __name__ == "__main__":
    train_pipeline()
    test_prediction()