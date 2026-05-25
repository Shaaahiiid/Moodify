import os
import math
import requests as http_requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

from src.data_fetch import fetch_user_profile, fetch_liked_songs, fetch_audio_features
from src.predict import predict_songs, FEATURE_COLUMNS, CLUSTER_LABELS

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "6d322602fdf447bd9ff2e7ad6ea41a9d")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET","e235bebea5a548469648bbe22678d7bb")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

CLUSTER_META = {
    0: {"name": "ENERGETIC / ROCK", "emoji": "🔥", "color": "#ef4444",
        "desc": "High energy tracks — loud, fast, and intense"},
    1: {"name": "AMBIENT / INSTRUMENTAL", "emoji": "🎹", "color": "#8b5cf6",
        "desc": "Quiet instrumental and ambient soundscapes"},
    2: {"name": "ACOUSTIC / CHILL", "emoji": "🌙", "color": "#38bdf8",
        "desc": "Acoustic & lo-fi vibes for study and focus sessions"},
    3: {"name": "DANCE / FEEL-GOOD", "emoji": "🕺", "color": "#1db954",
        "desc": "Upbeat, danceable tracks that lift your mood"},
    4: {"name": "LIVE / SPOKEN", "emoji": "🎤", "color": "#f59e0b",
        "desc": "Live recordings, podcasts, and spoken word"},
}


@app.route("/")
def index():
    return send_from_directory(".", "moodify.html")


@app.route("/callback")
def callback():
    code = request.args.get("code")
    error = request.args.get("error")

    if error:
        print(f"[callback] Spotify returned error: {error}")
        return redirect(f"/#error={error}")
    if not code:
        return redirect("/#error=no_code")

    redirect_uri = request.url_root.rstrip("/") + "/callback"
    print(f"[callback] Exchanging code for token, redirect_uri={redirect_uri}")

    token_resp = http_requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    if token_resp.status_code != 200:
        print(f"[callback] Token exchange FAILED: {token_resp.status_code} - {token_resp.text}")
        return redirect(f"/#error=token_exchange_failed")

    token_data = token_resp.json()
    access_token = token_data.get("access_token", "")
    print(f"[callback] Token exchange SUCCESS, token length={len(access_token)}")

    return redirect(f"/#access_token={access_token}")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "access_token" not in data:
        return jsonify({"error": "access_token is required"}), 400

    token = data["access_token"]

    try:
        user = fetch_user_profile(token)

        tracks = fetch_liked_songs(token, limit=1000)
        if not tracks:
            return jsonify({"error": "No liked songs found"}), 404

        track_ids = [t["id"] for t in tracks]
        track_meta = [{"name": t["name"], "artist": t["artist"]} for t in tracks]
        features_list = fetch_audio_features(token, track_ids, track_meta)

        rows = []
        valid_tracks = []
        skipped_count = 0
        skipped_tracks = []
        for track, features in zip(tracks, features_list):
            if features is None:
                skipped_count += 1
                skipped_tracks.append({"name": track["name"], "artist": track["artist"]})
                continue
            row = {col: features.get(col, 0) for col in FEATURE_COLUMNS}
            rows.append(row)
            track["bpm"] = round(features.get("tempo", 0))
            track["art"] = "🎵"
            valid_tracks.append(track)

        if not rows:
            return jsonify({"error": "Could not fetch audio features"}), 500

        if skipped_count:
            print(f"[analyze] {skipped_count}/{len(tracks)} songs skipped (no audio features found)")

        features_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

        clusters = predict_songs(features_df)

        cluster_data = {}
        for track, cluster_id, (_, feat_row) in zip(
            valid_tracks, clusters, features_df.iterrows()
        ):
            if cluster_id not in cluster_data:
                meta = CLUSTER_META.get(cluster_id, {
                    "name": f"CLUSTER {cluster_id}",
                    "emoji": "🎶",
                    "color": "#1db954",
                    "desc": "Auto-grouped by ML",
                })
                cluster_data[cluster_id] = {
                    "id": int(cluster_id),
                    "name": meta["name"],
                    "emoji": meta["emoji"],
                    "color": meta["color"],
                    "desc": meta["desc"],
                    "tracks": [],
                    "_energy": 0, "_dance": 0, "_valence": 0,
                    "_tempo": 0, "_acoustic": 0,
                }
                
            cd = cluster_data[cluster_id]
            cd["tracks"].append({
                "id": track["id"],
                "name": track["name"],
                "artist": track["artist"],
                "bpm": track["bpm"],
                "art": track["art"],
            })
            cd["_energy"] += feat_row["energy"]
            cd["_dance"] += feat_row["danceability"]
            cd["_valence"] += feat_row["valence"]
            cd["_tempo"] += feat_row["tempo"]
            cd["_acoustic"] += feat_row["acousticness"]

        result_clusters = []
        for cid in sorted(cluster_data.keys()):
            cd = cluster_data[cid]
            n = len(cd["tracks"])
            cd["songs"] = n
            cd["features"] = {
                "energy": round(cd["_energy"] / n * 100) if n else 0,
                "dance": round(cd["_dance"] / n * 100) if n else 0,
                "valence": round(cd["_valence"] / n * 100) if n else 0,
                "tempo": round(cd["_tempo"] / n) if n else 0,
                "acoustic": round(cd["_acoustic"] / n * 100) if n else 0,
            }
            for k in ["_energy", "_dance", "_valence", "_tempo", "_acoustic"]:
                del cd[k]
            result_clusters.append(cd)

        total = len(valid_tracks)
        top_cluster = max(result_clusters, key=lambda c: c["songs"])
        avg_energy = sum(features_df["energy"]) / len(features_df) * 100
        avg_dance = sum(features_df["danceability"]) / len(features_df) * 100

        insights = [
            {
                "icon": "🔥",
                "title": "Your energy profile",
                "text": f"Your average energy score is {avg_energy:.0f}/100. "
                        + ("You're definitely an energy chaser!" if avg_energy > 60
                           else "You lean towards mellower tracks."),
            },
            {
                "icon": "🧠",
                "title": "Genre labels don't define you",
                "text": "Our ML found patterns human curation would miss — "
                        "clusters are based on how songs actually sound, not genre tags.",
            },
            {
                "icon": "💃",
                "title": "Danceability check",
                "text": f"Your average danceability is {avg_dance:.0f}/100 "
                        f"({'above' if avg_dance > 52 else 'below'} the Spotify average of 52).",
            },
            {
                "icon": "🎯",
                "title": f"{top_cluster['name']} is your dominant mood",
                "text": f"Your largest cluster has {top_cluster['songs']} songs "
                        f"out of {total} total — that's {top_cluster['songs']/total*100:.0f}% of your library.",
            },
        ]

        radar_values = [
            round(avg_energy / 100, 2),
            round(avg_dance / 100, 2),
            round(sum(features_df["valence"]) / len(features_df), 2),
            round(min(sum(features_df["tempo"]) / len(features_df) / 200, 1.0), 2),
            round(sum(features_df["acousticness"]) / len(features_df), 2),
        ]

        return jsonify({
            "user": user,
            "total_songs": total,
            "skipped_songs": skipped_count,
            "skipped_tracks": skipped_tracks,
            "clusters": result_clusters,
            "insights": insights,
            "radar_values": radar_values,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

