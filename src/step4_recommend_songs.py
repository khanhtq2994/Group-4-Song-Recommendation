#!/usr/bin/env python3
"""
Bước 4 - Gợi ý các bài hát tương tự với một bài do người dùng nhập vào.

Tương ứng Box 11.7 trong sách và sơ đồ Figure 11.10:
    bài đầu vào -> tra cụm của nó -> chỉ so sánh với các bài cùng cụm
                -> tính khoảng cách -> lấy top N bài gần nhất.
Nhờ bước gom cụm, không gian tìm kiếm giảm ~k lần nên có thể chạy gần thời gian thực.

Khác biệt so với sách: người dùng truyền song-id (hoặc tên bài) qua tham số dòng
lệnh thay vì lặp qua toàn bộ dataset, và mặc định dùng khoảng cách Euclid trên
không gian đã chuẩn hoá. Muốn dùng đúng công thức của sách thì thêm --metric book.

Cách chạy:
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
    parser = argparse.ArgumentParser(description="Gợi ý bài hát tương tự")
    parser.add_argument("--clustered", default="data/songs_clustered",
                        help="Thư mục kết quả của bước 3")
    parser.add_argument("--model", default="data/kmeans_model.p")
    parser.add_argument("--song-id", default="", help="Song ID của bài đầu vào")
    parser.add_argument("--title", default="", help="Tìm bài đầu vào theo tên (gần đúng)")
    parser.add_argument("--random", action="store_true",
                        help="Chọn ngẫu nhiên một bài làm đầu vào")
    parser.add_argument("--top", type=int, default=common.DEFAULT_TOP_N)
    parser.add_argument("--metric", choices=["euclidean", "book"], default="euclidean")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not (args.song_id or args.title or args.random):
        sys.exit("Cần một trong các tham số: --song-id, --title hoặc --random")
    if not os.path.exists(args.model):
        sys.exit("Chưa có model '%s'. Hãy chạy bước 3 trước." % args.model)

    model = common.load_model(args.model)
    centers, mean, std = model["centers"], model["mean"], model["std"]
    feature_cols = model.get("feature_cols", common.FEATURE_COLS)

    sc = common.make_spark_context("SongRecommendation")
    try:
        rows = (sc.textFile("file://" + os.path.abspath(args.clustered))
                  .map(common.parse_csv_line)
                  .filter(lambda f: common.is_data_row(f, len(common.CLUSTERED_COLUMNS)))
                  .cache())

        # --- Tìm bài hát đầu vào -------------------------------------------
        if args.song_id:
            matches = rows.filter(lambda f: f[IDX["song_id"]] == args.song_id).take(1)
        elif args.title:
            needle = args.title.lower()
            matches = rows.filter(
                lambda f: needle in f[IDX["song_name"]].lower()).take(1)
        else:
            matches = rows.takeSample(False, 1, seed=random.randint(0, 10 ** 6))
        if not matches:
            sys.exit("Không tìm thấy bài hát đầu vào trong dataset.")
        seed_song = matches[0]

        cluster_id = common.to_int(seed_song[CLUSTER_IDX])
        center = centers[cluster_id]
        seed_vec = common.standardize(
            common.feature_vector(seed_song, feature_cols), mean, std)

        info = describe(seed_song)
        print("Bài hát đầu vào : %s - %s (%s)"
              % (info["song_name"], info["artist_name"], info["song_id"]))
        print("Cụm tìm được    : %d" % cluster_id)
        print("Tâm cụm (z-score): %s" % [round(x, 3) for x in center])

        # --- So sánh với các bài khác trong cùng cụm ------------------------
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
        print("Số bài cùng cụm  : %d (toàn bộ dataset: %d)" % (n_candidates, rows.count()))

        print("\n%d bài hát tương tự nhất (metric=%s):" % (len(top), args.metric))
        print("  %-4s %-38s %-28s %-6s %s" % ("#", "Bài hát", "Nghệ sĩ", "Năm", "Khoảng cách"))
        similar = []
        for rank, (dist, fields) in enumerate(top, 1):
            item = describe(fields)
            item["distance"] = round(dist, 6)
            similar.append(item)
            print("  %-4d %-38s %-28s %-6s %.6f"
                  % (rank, item["song_name"][:38], item["artist_name"][:28],
                     item["year"] or "-", dist))

        # cf. dict `post` trong Box 11.7 (sách đẩy kết quả này vào MongoDB)
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
            print("\nĐã ghi kết quả vào %s" % args.output_json)
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
