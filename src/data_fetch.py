import re
import unicodedata
import pandas as pd
import spotipy

_dataset_cache = None
_name_index_cache = None


def _normalize(text):
    """Normalize a track/artist name for fuzzy matching.
    Strips remaster/remix/feat suffixes, accents, and special characters."""
    text = str(text).strip().lower()
    # Decompose unicode accents (Beyoncé -> Beyonce)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Strip common suffixes: " - Remastered 2011", " (feat. X)", " (Deluxe)", etc.
    text = re.sub(r"\s*[-–—]\s*(remaster(ed)?|remix(ed)?|deluxe|bonus|live|acoustic|radio|extended|original|anniversary|mono|stereo).*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\((feat\.|ft\.|featuring|with|remaster(ed)?|deluxe|bonus|live|acoustic|radio|remix|extended|original|anniversary|mono|stereo)[^)]*\)", "", text, flags=re.IGNORECASE)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_dataset():
    global _dataset_cache, _name_index_cache
    if _dataset_cache is None:
        df = pd.read_csv("data/raw/spotify_data_merged.csv")
        _dataset_cache = df.set_index("track_id")

        _name_index_cache = {}
        for _, row in df.iterrows():
            track = _normalize(row.get("track_name", ""))
            if track and track not in _name_index_cache:
                _name_index_cache[track] = row

        print(f"[data_fetch] Loaded {len(_dataset_cache)} tracks (ID index) + {len(_name_index_cache)} unique track names")
    return _dataset_cache


def fetch_user_profile(token):
    sp = spotipy.Spotify(auth=token)
    me = sp.current_user()
    return {
        "name": me.get("display_name", "User"),
        "id": me.get("id", "unknown"),
        "initials": (me.get("display_name") or "U")[0].upper(),
    }


def fetch_liked_songs(token, limit=1000):
    sp = spotipy.Spotify(auth=token)
    tracks = []
    offset = 0
    batch_size = 50

    while offset < limit:
        results = sp.current_user_saved_tracks(
            limit=min(batch_size, limit - offset), offset=offset
        )
        items = results.get("items", [])
        if not items:
            break

        for item in items:
            track = item.get("track")
            if track and track.get("id"):
                tracks.append(
                    {
                        "id": track["id"],
                        "name": track.get("name", "Unknown"),
                        "artist": (
                            track.get("artists", [{}])[0].get("name", "Unknown")
                        ),
                    }
                )
        offset += batch_size
        print(f"[data_fetch] Fetched {len(tracks)} liked songs so far...")

    print(f"[data_fetch] Total liked songs fetched: {len(tracks)}")
    return tracks


def fetch_audio_features(token, track_ids, track_names_artists=None):
    try:
        sp = spotipy.Spotify(auth=token)
        all_features = []
        batch_size = 100

        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i : i + batch_size]
            features = sp.audio_features(batch)
            all_features.extend(features)

        print(f"[data_fetch] Fetched audio features from Spotify API for {len(all_features)} tracks")
        return all_features

    except Exception as e:
        print(f"[data_fetch] Spotify audio-features API failed: {e}")
        print("[data_fetch] Falling back to Kaggle dataset lookup...")
        return lookup_features_from_dataset(track_ids, track_names_artists)


def lookup_features_from_dataset(track_ids, track_names_artists=None):
    dataset = _load_dataset()
    results = []
    found_by_id = 0
    found_by_name = 0
    skipped_names = []

    feature_keys = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]

    for i, tid in enumerate(track_ids):
        row = None

        # Tier 1: exact track ID match
        if tid in dataset.index:
            row = dataset.loc[tid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            found_by_id += 1

        # Tier 2: track name match (no artist — avoids multi-artist mismatch)
        elif track_names_artists and i < len(track_names_artists):
            name = _normalize(track_names_artists[i]["name"])
            if name and name in _name_index_cache:
                row = _name_index_cache[name]
                found_by_name += 1

        if row is not None:
            results.append({k: row.get(k, 0) for k in feature_keys})
        else:
            results.append(None)
            if track_names_artists and i < len(track_names_artists):
                skipped_names.append(
                    f"{track_names_artists[i]['name']} — {track_names_artists[i]['artist']}"
                )

    total = found_by_id + found_by_name
    skipped = len(track_ids) - total
    print(f"[data_fetch] Dataset lookup: {total}/{len(track_ids)} matched "
          f"({found_by_id} by ID, {found_by_name} by track name), "
          f"{skipped} skipped")
    if skipped_names:
        print(f"[data_fetch] Sample skipped: {skipped_names[:5]}")
    return results
