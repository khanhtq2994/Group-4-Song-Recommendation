#!/usr/bin/env python3
"""
Bước 1 - Đọc các file HDF5 của Million Song Dataset (subset) và xuất metadata ra CSV.

Tương ứng Box 11.4 trong sách. Khác biệt: sách dùng module `hdf5_getters` của MSD
và chỉ đọc một file "input.h5"; ở đây đọc thẳng bằng h5py (không cần file phụ) và
duyệt đệ quy toàn bộ thư mục dataset, đồng thời hỗ trợ file summary gộp
(subset_msd_summary_file.h5 - chứa cả 10.000 bài trong một file, đọc nhanh hơn nhiều).

Cách chạy:
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
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("Thiếu thư viện h5py. Cài bằng: pip install h5py")


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
    """Sinh ra từng bài hát trong một file .h5 (file lẻ hoặc file summary gộp)."""
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
                        help="Thư mục MillionSongSubset hoặc một file .h5")
    parser.add_argument("--output", default="data/songs.csv", help="File CSV đầu ra")
    parser.add_argument("--limit", type=int, default=0,
                        help="Chỉ lấy N bài đầu tiên (0 = lấy hết), tiện để thử nhanh")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit("Không tìm thấy '%s'. Xem README để tải dataset." % args.input)

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
                        print("  ... đã đọc %d bài" % total)
                    if args.limit and total >= args.limit:
                        raise StopIteration
            except StopIteration:
                break
            except (OSError, KeyError) as err:
                failed += 1
                print("  [bỏ qua] %s: %s" % (path, err), file=sys.stderr)

    print("Đã ghi %d bài hát vào %s (%d file lỗi)." % (total, args.output, failed))
    if total == 0:
        sys.exit("Không đọc được bài nào - kiểm tra lại đường dẫn --input.")


if __name__ == "__main__":
    main()
