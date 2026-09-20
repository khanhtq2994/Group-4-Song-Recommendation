#!/usr/bin/env python3
"""
Bước 3 - Gom cụm các bài hát bằng K-Means (Spark MLlib).

Tương ứng Box 11.6 trong sách: gom 10 cụm theo đặc trưng âm thanh, ghi tâm cụm ra
file pickle và ghi lại dataset kèm số hiệu cụm của từng bài.

Khác biệt so với sách:
  * Tham số `runs` đã bị bỏ từ Spark 2.0 -> không dùng nữa.
  * Chuẩn hoá z-score trước khi gom cụm (loudness dB, tempo BPM và các confidence
    có thang đo rất lệch nhau; không chuẩn hoá thì tempo chi phối khoảng cách).
  * Lưu thêm mean/std vào pickle để bước 4 dùng lại đúng không gian đặc trưng.

Cách chạy:
    python3 src/step3_cluster_songs.py --input data/songs.csv --k 10 --overwrite
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

from pyspark.mllib.clustering import KMeans


def main():
    parser = argparse.ArgumentParser(description="Gom cụm bài hát bằng K-Means")
    parser.add_argument("--input", default="data/songs.csv")
    parser.add_argument("--output", default="data/songs_clustered",
                        help="Thư mục Spark ghi dataset đã gán cụm")
    parser.add_argument("--model", default="data/kmeans_model.p",
                        help="File pickle chứa tâm cụm (cf. cluster_centers.p)")
    parser.add_argument("--k", type=int, default=common.DEFAULT_K)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    common.prepare_output_dir(args.output, args.overwrite)
    sc = common.make_spark_context("SongClustering")
    try:
        rows = (sc.textFile("file://" + os.path.abspath(args.input))
                  .map(common.parse_csv_line)
                  .filter(common.is_data_row)
                  .filter(lambda f: common.valid_vector(common.feature_vector(f)))
                  .cache())

        features = rows.map(common.feature_vector).cache()
        mean, std, n = common.column_stats(features)
        print("Số bài hợp lệ: %d" % n)
        for col, m, s in zip(common.FEATURE_COLS, mean, std):
            print("  %-18s mean=%10.4f  std=%10.4f" % (col, m, s))

        scaled = features.map(lambda v: common.standardize(v, mean, std)).cache()

        # cf. KMeans.train(Parseddata, 10, maxIterations=100, ...) trong Box 11.6
        model = KMeans.train(scaled, args.k,
                             maxIterations=args.max_iterations,
                             initializationMode="k-means||",
                             seed=args.seed)
        centers = [[float(x) for x in c] for c in model.clusterCenters]
        wssse = model.computeCost(scaled)
        print("\nWSSSE (tổng bình phương khoảng cách tới tâm cụm) = %.4f" % wssse)

        # Lưu tâm cụm + tham số chuẩn hoá (sách: pickle.dump(cc, ...))
        common.save_model(args.model, {
            "feature_cols": common.FEATURE_COLS,
            "k": args.k,
            "centers": centers,
            "mean": mean,
            "std": std,
            "wssse": wssse,
            "n_songs": n,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        print("Đã lưu model vào %s" % args.model)

        # Gán cụm cho từng bài rồi ghi lại CSV kèm cột 'cluster'
        bc = sc.broadcast({"centers": centers, "mean": mean, "std": std})

        def assign(fields):
            p = bc.value
            vec = common.standardize(common.feature_vector(fields), p["mean"], p["std"])
            return list(fields) + [common.closest_point(vec, p["centers"])]

        assigned = rows.map(assign).cache()
        assigned.map(common.to_csv_line).saveAsTextFile("file://" + os.path.abspath(args.output))
        print("Đã ghi dataset đã gán cụm vào %s/" % args.output)

        print("\nKích thước từng cụm:")
        sizes = assigned.map(lambda f: (f[-1], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
        for cid in sorted(sizes):
            center = [round(x, 3) for x in centers[cid]]
            print("  cụm %2d: %5d bài   tâm(z-score)=%s" % (cid, sizes[cid], center))
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
