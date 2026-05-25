import pickle
import pandas as pd

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

CLUSTER_LABELS = {
    0: "Energetic / Rock",
    1: "Ambient / Instrumental",
    2: "Acoustic / Chill",
    3: "Dance / Feel-Good",
    4: "Live / Spoken"
}


def load_model(path="models/kmeans_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def load_scaler(path="models/scaler.pkl"):
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def predict_song(features):
    model = load_model()
    scaler = load_scaler()

    features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    scaled = scaler.transform(features_df)

    cluster = model.predict(scaled)[0]

    return CLUSTER_LABELS[cluster]


def predict_songs(features_df):
    model = load_model()
    scaler = load_scaler()

    scaled = scaler.transform(features_df[FEATURE_COLUMNS])
    clusters = model.predict(scaled)

    return clusters.tolist()


if __name__ == "__main__":
    features = [0.75, 0.7, -5.5, 0.08, 0.12, 0.0, 0.15, 0.6, 120]
    print(predict_song(features))