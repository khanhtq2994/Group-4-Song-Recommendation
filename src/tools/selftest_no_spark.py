#!/usr/bin/env python3
"""
Run the logic of steps 3 and 4 using plain Python (no Spark/Java needed) on the
sample CSV file - used to quickly test the computation part before running
the real Spark job.

    python3 src/tools/make_demo_csv.py --output data/songs_demo.csv --n 500
    python3 src/tools/selftest_no_spark.py --input data/songs_demo.csv
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def kmeans(points, k, iterations=50, seed=42):
    """Minimal K-Means (replaces MLlib) for self-testing."""
    rnd = random.Random(seed)
    centers = [list(p) for p in rnd.sample(points, k)]
    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for p in points:
            groups[common.closest_point(p, centers)].append(p)
        moved = 0.0
        for i, group in enumerate(groups):
            if not group:
                continue
            new_center = [sum(col) / len(group) for col in zip(*group)]
            moved += common.euclidean_distance(new_center, centers[i])
            centers[i] = new_center
        if moved < 1e-6:
            break
    return centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/songs_demo.csv")
    parser.add_argument("--k", type=int, default=common.DEFAULT_K)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [common.parse_csv_line(line) for line in f]
    rows = [r for r in rows if common.is_data_row(r)
            and common.valid_vector(common.feature_vector(r))]
    print("Read %d valid songs" % len(rows))
    assert rows, "No data available"

    features = [common.feature_vector(r) for r in rows]
    dim = len(features[0])
    n = len(features)
    mean = [sum(v[i] for v in features) / n for i in range(dim)]
    std = [(sum((v[i] - mean[i]) ** 2 for v in features) / n) ** 0.5 for i in range(dim)]
    scaled = [common.standardize(v, mean, std) for v in features]

    centers = kmeans(scaled, args.k)
    labels = [common.closest_point(v, centers) for v in scaled]
    sizes = {}
    for c in labels:
        sizes[c] = sizes.get(c, 0) + 1
    print("Cluster sizes:", dict(sorted(sizes.items())))
    assert sum(sizes.values()) == n

    seed_i = 0
    seed_cluster, seed_vec = labels[seed_i], scaled[seed_i]
    print("\nInput song: %s - %s (cluster %d)"
          % (rows[seed_i][common.IDX["song_name"]],
             rows[seed_i][common.IDX["artist_name"]], seed_cluster))

    for metric in ("euclidean", "book"):
        scored = [(common.song_distance(scaled[i], seed_vec, centers[seed_cluster], metric),
                   rows[i][common.IDX["song_name"]], rows[i][common.IDX["artist_name"]])
                  for i in range(n) if labels[i] == seed_cluster and i != seed_i]
        scored.sort(key=lambda x: x[0])
        print("  metric=%-9s ->" % metric)
        for dist, name, artist in scored[:args.top]:
            print("      %.6f  %-30s %s" % (dist, name[:30], artist[:25]))

    # test CSV parser with song names containing commas
    line = common.to_csv_line(["a", "Artist, ft. B", "c", 'Song "hit", live version'] + [0] * 8)
    assert common.parse_csv_line(line)[3] == 'Song "hit", live version'
    print("\nSelf-test OK.")


if __name__ == "__main__":
    main()
