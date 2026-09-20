#!/usr/bin/env python3
"""
Sinh một file CSV giả lập đúng định dạng của bước 1, để chạy thử bước 2-3-4 khi
chưa tải xong Million Song Subset (2 GB).

    python3 src/tools/make_demo_csv.py --output data/songs_demo.csv --n 2000
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

GENRES = [  # (tên, loudness, tempo, key_conf, mode_conf)
    ("Ballad", -14.0, 72.0, 0.45, 0.50),
    ("Pop", -8.0, 118.0, 0.60, 0.55),
    ("Dance", -6.0, 128.0, 0.70, 0.60),
    ("Rock", -5.0, 140.0, 0.55, 0.45),
    ("Metal", -4.0, 165.0, 0.40, 0.40),
]


def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu mẫu")
    parser.add_argument("--output", default="data/songs_demo.csv")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(common.COLUMNS)
        for i in range(args.n):
            genre, loud, tempo, keyc, modec = GENRES[i % len(GENRES)]
            artist_no = i % 120
            # song_hotttnesss bị thiếu ở ~30% bài, giống dữ liệu thật
            hot = "nan" if rnd.random() < 0.3 else round(rnd.random(), 4)
            writer.writerow([
                "ARDEMO%06d" % artist_no,
                "Artist %03d (%s)" % (artist_no, genre),
                "SODEMO%06d" % i,
                "%s song #%d, live" % (genre, i),   # có dấu phẩy để test parser CSV
                round(rnd.gauss(loud, 1.5), 3),
                hot,
                round(rnd.gauss(tempo, 6.0), 3),
                round(min(max(rnd.gauss(keyc, 0.08), 0.0), 1.0), 3),
                round(min(max(rnd.gauss(modec, 0.08), 0.0), 1.0), 3),
                rnd.choice([0, 1990, 1993, 1995, 1998, 2001, 2005, 2008]),
                round(rnd.random(), 4),
                round(rnd.gauss(230, 40), 3),
            ])
    print("Đã sinh %d bài hát mẫu vào %s" % (args.n, args.output))


if __name__ == "__main__":
    main()
