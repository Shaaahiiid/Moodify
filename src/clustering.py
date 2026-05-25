from sklearn.cluster import KMeans
import pickle


def train_kmeans(X_scaled, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    return model


def save_model(model, path="../models/kmeans_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)