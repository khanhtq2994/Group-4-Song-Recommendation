#!/usr/bin/env python3
"""
Step 4 - Suggest songs similar to a song entered by the user.

Corresponding to Box 11.7 in the book and Figure 11.10:
    Input song -> find its cluster -> only compare with songs in the same cluster
                -> calculate distance -> get top N nearest songs.
Thanks to the clustering step, the search space is reduced ~k times, enabling near real-time performance.

Difference from the book: the user passes song-id (or song name) via command line
parameter instead of iterating through the entire dataset, and uses Euclidean distance
on the normalized space by default. To use the book's exact formula, add --metric book.

How to run:
    python3 src/step4_recommend_songs.py --song-id SOICLQB12A8C13637C
    python3 src/step4_recommend_songs.py --title "Exodus" --top 10
"""

import argparse
import datetime
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

IDX = common.IDX
CLUSTER_IDX = common.CLUSTERED_COLUMNS.index(common.CLUSTER_COL)


def describe(fields):
    return {
        "song_id": fields[IDX["song_id"]],
        "song_name": fields[IDX["song_name"]],
        "artist_name": fields[IDX["artist_name"]],
        "artist_id": fields[IDX["artist_id"]],
        "year": common.to_int(fields[IDX["year"]]),
        "tempo": round(common.to_float(fields[IDX["tempo"]]), 2),
        "loudness": round(common.to_float(fields[IDX["loudness"]]), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Suggest similar songs")
    parser.add_argument("--clustered", default="data/songs_clustered",
                        help="Output directory from step 3")
    parser.add_argument("--model", default="data/kmeans_model.p")
    parser.add_argument("--song-id", default="", help="Song ID of the input song")
    parser.add_argument("--title", default="", help="Find input song by name (approximate)")
    parser.add_argument("--random", action="store_true",
                        help="Randomly select an input song")
    parser.add_argument("--top", type=int, default=common.DEFAULT_TOP_N)
    parser.add_argument("--metric", choices=["euclidean", "book"], default="euclidean")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not (args.song_id or args.title or args.random):
        sys.exit("Need one of the parameters: --song-id, --title or --random")
    if not os.path.exists(args.model):
        sys.exit("Model '%s' does not exist. Please run step 3 first." % args.model)

    model = common.load_model(args.model)
    centers, mean, std = model["centers"], model["mean"], model["std"]
    feature_cols = model.get("feature_cols", common.FEATURE_COLS)

    sc = common.make_spark_context("SongRecommendation")
    try:
        rows = (sc.textFile("file://" + os.path.abspath(args.clustered))
                  .map(common.parse_csv_line)
                  .filter(lambda f: common.is_data_row(f, len(common.CLUSTERED_COLUMNS)))
                  .cache())

        # --- Find the input song -------------------------------------------
        if args.song_id:
            matches = rows.filter(lambda f: f[IDX["song_id"]] == args.song_id).take(1)
        elif args.title:
            needle = args.title.lower()
            matches = rows.filter(
                lambda f: needle in f[IDX["song_name"]].lower()).take(1)
        else:
            matches = rows.takeSample(False, 1, seed=random.randint(0, 10 ** 6))
        if not matches:
            sys.exit("Could not find the input song in the dataset.")
        seed_song = matches[0]

        cluster_id = common.to_int(seed_song[CLUSTER_IDX])
        center = centers[cluster_id]
        seed_vec = common.standardize(
            common.feature_vector(seed_song, feature_cols), mean, std)

        info = describe(seed_song)
        print("Input song: %s - %s (%s)"
              % (info["song_name"], info["artist_name"], info["song_id"]))
        print("Found cluster: %d" % cluster_id)
        print("Cluster center (z-score): %s" % [round(x, 3) for x in center])

        # --- Compare with other songs in the same cluster ------------------------
        bc = sc.broadcast({"mean": mean, "std": std, "center": center,
                           "seed": seed_vec, "cols": feature_cols,
                           "metric": args.metric})

        def score(fields):
            p = bc.value
            vec = common.standardize(
                common.feature_vector(fields, p["cols"]), p["mean"], p["std"])
            return (common.song_distance(vec, p["seed"], p["center"], p["metric"]), fields)

        candidates = rows.filter(
            lambda f: common.to_int(f[CLUSTER_IDX]) == cluster_id
        ).filter(
            lambda f: f[IDX["song_id"]] != seed_song[IDX["song_id"]]
        ).filter(
            lambda f: common.valid_vector(common.feature_vector(f, feature_cols))
        )

        n_candidates = candidates.count()
        top = candidates.map(score).takeOrdered(args.top, key=lambda x: x[0])
        print("Number of songs in the same cluster: %d (total dataset: %d)" % (n_candidates, rows.count()))

        print("\n%d similar songs (metric=%s):" % (len(top), args.metric))
        print("  %-4s %-38s %-28s %-6s %s" % ("#", "Song", "Artist", "Year", "Distance"))
        similar = []
        for rank, (dist, fields) in enumerate(top, 1):
            item = describe(fields)
            item["distance"] = round(dist, 6)
            similar.append(item)
            print("  %-4d %-38s %-28s %-6s %.6f"
                  % (rank, item["song_name"][:38], item["artist_name"][:28],
                     item["year"] or "-", dist))

        # cf. dict `post` in Box 11.7 (the book pushes these results to MongoDB)
        post = {
            "name": info["song_name"],
            "artist": info["artist_name"],
            "song_id": info["song_id"],
            "year": info["year"],
            "cluster": cluster_id,
            "metric": args.metric,
            "date": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "similar": similar,
        }
        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(post, f, ensure_ascii=False, indent=2)
            print("\nResults have been saved to %s" % args.output_json)
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
