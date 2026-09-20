"""
Cấu hình & tiện ích dùng chung cho case study "Song Recommendation System".

Tham khảo: Bahga & Madisetti, "Big Data Science & Analytics: A Hands-On Approach",
mục 11.3 (Box 11.4 - Box 11.7). Code ở đây được viết lại cho Python 3 + Spark 3.x.
"""

import csv
import io
import math
import os
import pickle
import shutil

# --------------------------------------------------------------------------
# Lược đồ dữ liệu
# --------------------------------------------------------------------------
# Thứ tự cột của file CSV sinh ra ở bước 1 (cf. Box 11.4).
# 10 cột đầu giữ đúng thứ tự như trong sách, 2 cột cuối được thêm vào để
# xếp hạng nghệ sĩ (artist_hotttnesss) và để hiển thị (duration).
COLUMNS = [
    "artist_id",          # 0
    "artist_name",        # 1
    "song_id",            # 2
    "song_name",          # 3
    "loudness",           # 4
    "song_hotttnesss",    # 5
    "tempo",              # 6
    "key_confidence",     # 7
    "mode_confidence",    # 8
    "year",               # 9
    "artist_hotttnesss",  # 10
    "duration",           # 11
]
IDX = {name: i for i, name in enumerate(COLUMNS)}

CLUSTER_COL = "cluster"                        # cột được thêm sau bước 3
CLUSTERED_COLUMNS = COLUMNS + [CLUSTER_COL]

# Bốn đặc trưng dùng để gom cụm. Sách dùng cột 4..7 tức
# (loudness, song_hotttnesss, tempo, key_confidence), nhưng song_hotttnesss bị
# NaN ở rất nhiều bài trong subset nên mặc định thay bằng mode_confidence.
# Muốn chạy đúng y như sách thì chỉ cần sửa danh sách này.
FEATURE_COLS = ["loudness", "tempo", "key_confidence", "mode_confidence"]

DEFAULT_K = 10          # số cụm (sách dùng 10)
DEFAULT_TOP_N = 10      # số bài gợi ý


# --------------------------------------------------------------------------
# Đọc / ghi CSV
# --------------------------------------------------------------------------
def parse_csv_line(line):
    """Tách một dòng CSV thành list các trường.

    Sách dùng `line.split(",")`, nhưng tên bài hát / nghệ sĩ có thể chứa dấu
    phẩy nên ở đây dùng module csv (bước 1 đã ghi ra CSV có quote đúng chuẩn).
    """
    try:
        return next(csv.reader(io.StringIO(line)))
    except StopIteration:
        return []


def to_csv_line(fields):
    """Ghép list các trường thành một dòng CSV (quote khi cần)."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(list(fields))
    return buf.getvalue()


def is_data_row(fields, min_cols=len(COLUMNS)):
    """Bỏ dòng trống, dòng thiếu cột và dòng header."""
    return len(fields) >= min_cols and fields[IDX["artist_id"]] != "artist_id"


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_number(value):
    """True nếu là số thực hợp lệ (không NaN, không vô cực)."""
    return isinstance(value, float) and not (math.isnan(value) or math.isinf(value))


# --------------------------------------------------------------------------
# Vector đặc trưng
# --------------------------------------------------------------------------
def feature_vector(fields, feature_cols=FEATURE_COLS):
    """Lấy vector đặc trưng của một bài hát từ list các trường CSV."""
    return [to_float(fields[IDX[col]]) for col in feature_cols]


def valid_vector(vector):
    """Loại các bài có đặc trưng NaN/inf (K-Means không xử lý được)."""
    return all(is_number(v) for v in vector)


def standardize(vector, mean, std):
    """Chuẩn hoá z-score. Các đặc trưng có thang đo rất khác nhau
    (loudness ~ -60..0 dB, tempo ~ 0..250 BPM, confidence ~ 0..1) nên nếu không
    chuẩn hoá thì tempo sẽ chi phối toàn bộ khoảng cách."""
    return [(v - m) / (s if s > 1e-12 else 1.0) for v, m, s in zip(vector, mean, std)]


def column_stats(features_rdd):
    """Trả về (mean, std, count) của từng chiều đặc trưng, tính trên RDD."""
    n = features_rdd.count()
    if n == 0:
        raise ValueError("Không có bài hát hợp lệ nào để tính thống kê.")
    dim = len(features_rdd.first())
    zero = [0.0] * dim

    def add(a, b):
        return [x + y for x, y in zip(a, b)]

    total = features_rdd.reduce(add)
    total_sq = features_rdd.map(lambda v: [x * x for x in v]).reduce(add)
    mean = [s / n for s in total]
    std = [math.sqrt(max(sq / n - m * m, 0.0)) for sq, m in zip(total_sq, mean)]
    return mean, std, n


# --------------------------------------------------------------------------
# Khoảng cách & gán cụm
# --------------------------------------------------------------------------
def closest_point(point, centers):
    """Chỉ số tâm cụm gần nhất (cf. hàm closestPoint trong Box 11.6)."""
    best_index, closest = 0, float("+inf")
    for i, center in enumerate(centers):
        dist = sum((p - c) ** 2 for p, c in zip(point, center))
        if dist < closest:
            closest, best_index = dist, i
    return best_index


def euclidean_distance(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def book_distance(p, user_point, center, eps=1e-6):
    """Công thức trong Box 11.7 (hàm euclid_dist của sách).

    Sách quy hai bài hát về "độ lệch tương đối so với tâm cụm" rồi kết hợp
    cosine và chênh lệch độ lớn:
        similarity = sqrt(cosine^2 + (|c| - |d|)^2)
    và sắp xếp tăng dần. Lưu ý: cosine ở đây đi ngược trực giác (hai vector
    cùng hướng cho cosine = 1 nên bị xem là "xa" hơn) - giữ lại để đối chiếu
    với sách, còn mặc định của chương trình là metric "euclidean".
    """
    def rel(vec):
        return [(v / (c if abs(c) > eps else eps)) - 1.0 for v, c in zip(vec, center)]

    c = rel(user_point)
    d = rel(p)
    dist1 = math.sqrt(sum(x * x for x in c))
    dist2 = math.sqrt(sum(x * x for x in d))
    if dist1 < eps or dist2 < eps:
        return abs(dist1 - dist2)
    cosine = sum(x * y for x, y in zip(c, d)) / dist1 / dist2
    return math.sqrt(cosine * cosine + (dist1 - dist2) ** 2)


def song_distance(p, user_point, center, metric="euclidean"):
    """Khoảng cách giữa bài ứng viên và bài người dùng nhập (càng nhỏ càng giống)."""
    if metric == "book":
        return book_distance(p, user_point, center)
    return euclidean_distance(p, user_point)


# --------------------------------------------------------------------------
# Spark & lưu model
# --------------------------------------------------------------------------
def make_spark_context(app_name, master="local[*]"):
    """Tạo SparkContext chạy được cả khi gọi `python3 script.py` lẫn `spark-submit`."""
    from pyspark import SparkConf, SparkContext

    conf = SparkConf().setAppName(app_name)
    if not conf.contains("spark.master"):
        conf.setMaster(master)
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")
    return sc


def save_model(path, model):
    """Lưu tâm cụm + tham số chuẩn hoá (cf. file cluster_centers.p của sách)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_output_dir(path, overwrite):
    """Spark từ chối ghi đè thư mục output -> xử lý trước cho rõ ràng."""
    if os.path.exists(path):
        if not overwrite:
            raise SystemExit(
                "Thư mục '%s' đã tồn tại. Xoá nó hoặc chạy lại với --overwrite." % path
            )
        shutil.rmtree(path)
