#!/usr/bin/env python3
"""
Step 1 - Read HDF5 files of Million Song Dataset (subset) and export metadata to CSV.

Corresponds to Box 11.4 in the book. Differences: the book uses the `hdf5_getters` module from MSD
and only reads one file "input.h5"; here we read directly with h5py (no auxiliary file needed) and
recursively traverse the entire dataset directory, while also supporting the merged summary file
(subset_msd_summary_file.h5 - contains all 10,000 songs in one file, much faster to read).

How to run:
    python3 src/step1_extract_h5_to_csv.py \
        --input data/MillionSongSubset --output data/songs.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

try:
    # pyrefly: ignore [missing-import]
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("Missing h5py library. Install by: pip install h5py")


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_songs(path):
    """Yield each song in an .h5 file (single file or merged summary file)."""
    with h5py.File(path, "r") as h5:
        meta = h5["metadata"]["songs"]          # artist_id, artist_name, title, ...
        analysis = h5["analysis"]["songs"]      # loudness, tempo, key_confidence, ...
        musicbrainz = h5["musicbrainz"]["songs"]  # year
        for i in range(len(meta)):
            m, a, b = meta[i], analysis[i], musicbrainz[i]
            yield [
                _text(m["artist_id"]),
                _text(m["artist_name"]),
                _text(m["song_id"]),
                _text(m["title"]),
                _num(a["loudness"]),
                _num(m["song_hotttnesss"]),
                _num(a["tempo"]),
                _num(a["key_confidence"]),
                _num(a["mode_confidence"]),
                common.to_int(b["year"]),
                _num(m["artist_hotttnesss"]),
                _num(a["duration"]),
            ]


def iter_h5_files(root):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".h5"):
                yield os.path.join(dirpath, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", required=True,
                        help="MillionSongSubset directory or an .h5 file")
    parser.add_argument("--output", default="data/songs.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only take the first N songs (0 = take all), convenient for quick testing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit("Could not find '%s'. See README to download the dataset." % args.input)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total, failed = 0, 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(common.COLUMNS)
        for path in iter_h5_files(args.input):
            try:
                for row in read_songs(path):
                    writer.writerow(row)
                    total += 1
                    if total % 1000 == 0:
                        print("  ... read %d songs" % total)
                    if args.limit and total >= args.limit:
                        raise StopIteration
            except StopIteration:
                break
            except (OSError, KeyError) as err:
                failed += 1
                print("  [skipped] %s: %s" % (path, err), file=sys.stderr)

    print("Wrote %d songs to %s (%d files failed)." % (total, args.output, failed))
    if total == 0:
        sys.exit("No songs read - check --input path.")


if __name__ == "__main__":
    main()
