#!/usr/bin/env python3
"""
Step 2 - Find top 10 "hottest" songs and top 10 "hottest" artists for each year using Spark.

Corresponds to Box 11.5 in the book. map / filter / distinct / sortByKey / take steps
retain the spirit of the book, only modified for Python 3 (no longer using .encode('utf-8'))
and splitting CSV lines using the csv module instead of split(",").

How to run:
    python3 src/step2_top_songs_artists.py --input data/songs.csv \
        --from-year 1990 --to-year 1999
    (or: spark-submit --py-files src/common.py src/step2_top_songs_artists.py ...)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

IDX = common.IDX


def main():
    parser = argparse.ArgumentParser(description="Top songs / artists by year")
    parser.add_argument("--input", default="data/songs.csv")
    parser.add_argument("--from-year", type=int, default=1990)
    parser.add_argument("--to-year", type=int, default=1999)
    parser.add_argument("--top", type=int, default=common.DEFAULT_TOP_N)
    parser.add_argument("--output-json", default="",
                        help="Output JSON file (optional)")
    args = parser.parse_args()

    sc = common.make_spark_context("TopSongsAndArtists")
    try:
        # step1 + step2 of the book: read file, split fields, remove empty lines / header
        rows = (sc.textFile("file://" + os.path.abspath(args.input))
                  .map(common.parse_csv_line)
                  .filter(common.is_data_row))

        in_range = rows.filter(
            lambda f: args.from_year <= common.to_int(f[IDX["year"]]) <= args.to_year
        ).cache()

        # step4: (artist_id, artist_name, artist_hotttnesss, year) + distinct()
        # an artist may have multiple songs, without distinct() it will be repeated when ranking.
        artists = in_range.map(lambda f: (
            f[IDX["artist_id"]],
            f[IDX["artist_name"]],
            common.to_float(f[IDX["artist_hotttnesss"]]),
            common.to_int(f[IDX["year"]]),
        )).distinct().filter(lambda t: common.is_number(t[2])).cache()

        # step8: (song_id, song_name, song_hotttnesss, year)
        songs = in_range.map(lambda f: (
            f[IDX["song_id"]],
            f[IDX["song_name"]],
            common.to_float(f[IDX["song_hotttnesss"]]),
            common.to_int(f[IDX["year"]]),
            f[IDX["artist_name"]],
        )).distinct().filter(lambda t: common.is_number(t[2])).cache()

        result = {}
        for year in range(args.from_year, args.to_year + 1):
            # step5/step6: swap key to hotttnesss then sortByKey descending + take(10)
            top_artists = (artists.filter(lambda t, y=year: t[3] == y)
                                  .map(lambda t: (t[2], (t[0], t[1])))
                                  .sortByKey(False)
                                  .take(args.top))
            # step9/step10: similarly for songs
            top_songs = (songs.filter(lambda t, y=year: t[3] == y)
                              .map(lambda t: (t[2], (t[0], t[1], t[4])))
                              .sortByKey(False)
                              .take(args.top))

            result[year] = {
                "top_artists": [{"artist_id": a[1][0], "artist_name": a[1][1],
                                 "hotttnesss": round(a[0], 4)} for a in top_artists],
                "top_songs": [{"song_id": s[1][0], "song_name": s[1][1],
                               "artist_name": s[1][2],
                               "hotttnesss": round(s[0], 4)} for s in top_songs],
            }

            print("\n=== Year %d ===" % year)
            if not top_artists and not top_songs:
                print("  (no data)")
                continue
            print("  Top %d artists:" % args.top)
            for rank, a in enumerate(top_artists, 1):
                print("    %2d. %-35s hotttnesss=%.4f" % (rank, a[1][1][:35], a[0]))
            print("  Top %d songs:" % args.top)
            for rank, s in enumerate(top_songs, 1):
                print("    %2d. %-35s | %-25s hotttnesss=%.4f"
                      % (rank, s[1][1][:35], s[1][2][:25], s[0]))

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("\nResults written to %s" % args.output_json)
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
