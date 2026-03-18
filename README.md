# 🎧 Moodify: Music Clustering using Audio Features

## 📌 Overview

Moodify is an end-to-end machine learning system that automatically organizes songs into meaningful groups based on their **audio characteristics** rather than traditional genres.

Instead of labels like *pop* or *rock*, the system groups songs based on how they **feel and behave**, such as:

* Energetic / Workout
* Dance / Club
* Acoustic / Calm
* Fast / EDM

The system supports **any Spotify user** through authentication and generates **personalized music groupings**.

---

## 🚀 Features

* Unsupervised learning using **KMeans clustering**
* Automatic grouping based on audio features
* PCA-based cluster visualization
* Predict cluster for **new songs**
* Spotify OAuth integration (for real users)
* End-to-end ML pipeline (data → model → output)

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Spotipy (Spotify API)
* Pickle (model serialization)

---

## 📂 Project Structure

```id="l9g1kp"
spotify-genre-analyzer/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── spotify_analysis.ipynb
│
├── src/
│   ├── data_fetch.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── visualization.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

### 1. User Authentication

* User logs in via Spotify OAuth
* Access token is generated securely

---

### 2. Data Collection

* Fetch liked songs using:

```text id="ds1j9r"
GET /v1/me/tracks
```

* Extract track IDs
* Fetch audio features:

```text id="n6rjfh"
GET /v1/audio-features/{id}
```

---

### 3. Data Processing

* Select relevant features:

  * danceability, energy, loudness, speechiness
  * acousticness, instrumentalness, liveness
  * valence, tempo
* Scale features using StandardScaler

---

### 4. Clustering

* Apply **KMeans algorithm**
* Determine optimal K using Elbow Method
* Assign cluster labels to songs

---

### 5. Interpretation

Clusters are labeled based on dominant features:

| Cluster | Meaning             |
| ------- | ------------------- |
| 0       | Energetic / Workout |
| 1       | Dance / Club        |
| 2       | Acoustic / Calm     |
| 3       | Fast / EDM          |

---

### 6. Visualization

* Use PCA to reduce dimensions
* Plot clusters in 2D space

---

### 7. Prediction Pipeline

The system can classify new songs:

```text id="gx5b1g"
new song features
↓
scaler.transform()
↓
model.predict()
↓
cluster label
```

---

## 🔐 Spotify Integration

### Authentication Flow

```text id="rly7oq"
User Login
↓
Spotify OAuth
↓
Access Token
↓
Fetch User Data
```

### Required Scope

```text id="t2c0fd"
user-library-read
```

### Data Pipeline

```text id="qg2wzv"
Login
↓
Fetch Liked Songs
↓
Extract Track IDs
↓
Fetch Audio Features
↓
Build Dataset
↓
Run ML Model
↓
Generate Playlists
```

---

## ▶️ Usage

### 1. Install Dependencies

```bash id="q7g6q9"
pip install -r requirements.txt
```

### 2. Run the Project

```bash id="bn5vqp"
python main.py
```

### 3. (Optional) Explore Notebook

```bash id="z6m2qp"
notebooks/spotify_analysis.ipynb
```

---

## 📊 Example Output

```text id="yzn2z2"
Energetic / Workout
-------------------
Believer - Imagine Dragons
Rise - Jonas Blue

Acoustic / Calm
---------------
Perfect - Ed Sheeran
lovely - Billie Eilish
```

---

## 🎯 Key Highlights

* Works for **any Spotify user**
* Uses **unsupervised learning (no labels required)**
* Builds **behavior-based playlists**
* Scalable and modular design
* Real-world ML system (not just a notebook)

---

## 🚧 Future Improvements

* Build UI using Streamlit or Flask
* Add real-time song input
* Auto-create playlists on Spotify
* Improve clustering with more features
* Add recommendation system

---

## 👤 Author

Shahid Rashid Shaikh

---

## 📌 Summary

Moodify demonstrates how machine learning can transform raw audio features into meaningful music experiences by automatically organizing songs into behavior-based playlists, making music discovery smarter and more personalized.
