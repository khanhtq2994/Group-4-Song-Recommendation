# Case Study: Song Recommendation System (Spark)

Bài thực hành trên lớp — môn *Big Data analytics algorithms* (lec05.2).
Code mẫu viết theo mục **11.3** của Bahga & Madisetti, *Big Data Science & Analytics:
A Hands-On Approach* (Box 11.4 → Box 11.7), cập nhật cho **Python 3 + Spark 3.x**.

Hệ thống dùng **content-based filtering**: không cần lịch sử nghe / rating của người
dùng, chỉ dựa trên đặc trưng âm thanh của bài hát trong Million Song Dataset.

## 1. Luồng xử lý (Figure 11.10)

```
   File .h5 (Million Song Subset)
              │  bước 1  (Box 11.4)
              ▼
   data/songs.csv   ──────────────► bước 2 (Box 11.5): top 10 bài / nghệ sĩ mỗi năm
              │  bước 3  (Box 11.6)
              ▼
   K-Means (k = 10) trên 4 đặc trưng
              │
              ├─► data/kmeans_model.p     (tâm cụm + tham số chuẩn hoá)
              └─► data/songs_clustered/   (dataset + cột `cluster`)
                          │  bước 4  (Box 11.7)
                          ▼
     bài hát đầu vào → tra cụm → tính khoảng cách với các bài CÙNG cụm
                     → sắp xếp tăng dần → top 10 bài tương tự
```

Ý tưởng cốt lõi: gom cụm trước để **thu hẹp không gian tìm kiếm** (chỉ so sánh trong
1/k dataset), nhờ đó việc gợi ý có thể chạy gần thời gian thực.

## 2. Cấu trúc thư mục

| File | Vai trò | Box trong sách |
|---|---|---|
| [src/common.py](src/common.py) | Lược đồ CSV, đặc trưng, hàm khoảng cách, tiện ích Spark | — |
| [src/step1_extract_h5_to_csv.py](src/step1_extract_h5_to_csv.py) | Đọc file HDF5 → CSV metadata | Box 11.4 |
| [src/step2_top_songs_artists.py](src/step2_top_songs_artists.py) | Top 10 bài hát / nghệ sĩ mỗi năm | Box 11.5 |
| [src/step3_cluster_songs.py](src/step3_cluster_songs.py) | Gom cụm K-Means bằng Spark MLlib | Box 11.6 |
| [src/step4_recommend_songs.py](src/step4_recommend_songs.py) | Gợi ý bài hát tương tự | Box 11.7 |
| [src/tools/make_demo_csv.py](src/tools/make_demo_csv.py) | Sinh dữ liệu giả để chạy thử khi chưa có dataset | — |
| [src/tools/selftest_no_spark.py](src/tools/selftest_no_spark.py) | Chạy lại logic bước 3–4 bằng Python thuần (không cần Spark) | — |

Định dạng CSV (bước 1 sinh ra, có dòng header):

```
artist_id, artist_name, song_id, song_name, loudness, song_hotttnesss,
tempo, key_confidence, mode_confidence, year, artist_hotttnesss, duration
```

10 cột đầu giữ đúng thứ tự như sách; 2 cột cuối thêm vào để xếp hạng nghệ sĩ và hiển thị.

## 3. Cài đặt

```bash
# JDK cho Spark (chọn 1 trong 2)
brew install openjdk@17        # macOS
sudo apt install openjdk-17-jdk # Ubuntu

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
java -version && python3 -c "import pyspark; print(pyspark.__version__)"
```

## 4. Tải dataset

Chỉ dùng **subset 10.000 bài** (≈2 GB) như yêu cầu của đề bài:

```bash
mkdir -p data && cd data
curl -O http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz
tar -xzf millionsongsubset.tar.gz          # → data/MillionSongSubset/
cd ..
```

Trang chính thức: <http://millionsongdataset.com/> (nếu link trên hỏng thì lấy link
`millionsongsubset.tar.gz` ở mục *Getting the dataset*). Nếu bản tải về có file gộp
`subset_msd_summary_file.h5` thì trỏ thẳng `--input` vào file đó sẽ nhanh hơn nhiều.

## 5. Chạy từng bước

```bash
# Bước 1 — HDF5 → CSV  (thêm --limit 500 để thử nhanh)
python3 src/step1_extract_h5_to_csv.py --input data/MillionSongSubset --output data/songs.csv

# Bước 2 — Top 10 bài hát / nghệ sĩ theo từng năm
python3 src/step2_top_songs_artists.py --input data/songs.csv \
    --from-year 1990 --to-year 1999 --output-json data/top_by_year.json

# Bước 3 — Gom cụm K-Means (k = 10)
python3 src/step3_cluster_songs.py --input data/songs.csv --k 10 --overwrite

# Bước 4 — Gợi ý 10 bài tương tự
python3 src/step4_recommend_songs.py --song-id SOICLQB12A8C13637C --top 10
python3 src/step4_recommend_songs.py --title "Exodus" --output-json data/recommend.json
python3 src/step4_recommend_songs.py --random --metric book   # dùng công thức của sách
```

Chạy trên cụm Spark thật thì thay bằng:

```bash
spark-submit --master yarn --py-files src/common.py src/step3_cluster_songs.py --input hdfs:///songs.csv
```

### Chạy thử khi chưa có dataset

```bash
python3 src/tools/make_demo_csv.py --output data/songs_demo.csv --n 2000
python3 src/tools/selftest_no_spark.py --input data/songs_demo.csv   # không cần Java
python3 src/step3_cluster_songs.py --input data/songs_demo.csv --overwrite
python3 src/step4_recommend_songs.py --random
```

## 6. Những chỗ sửa so với code trong sách (nên nêu trong báo cáo)

| Sách (2016, Python 2 / Spark 1.x) | Code này | Lý do |
|---|---|---|
| `print x`, `line.encode('utf-8')` | cú pháp Python 3 | sách viết cho Python 2 |
| `KMeans.train(..., runs=100)` | bỏ `runs` | tham số này đã bị gỡ từ Spark 2.0 |
| `line.split(",")` | `csv.reader` | tên bài hát/nghệ sĩ có dấu phẩy → lệch cột |
| Gom cụm trên giá trị gốc | chuẩn hoá z-score trước khi gom cụm | loudness (dB), tempo (BPM), confidence (0–1) lệch thang đo, nếu không chuẩn hoá thì tempo chi phối toàn bộ khoảng cách |
| Đặc trưng: loudness, song_hotttnesss, tempo, key_confidence | loudness, tempo, key_confidence, mode_confidence | `song_hotttnesss` bị NaN ở rất nhiều bài trong subset (sửa `FEATURE_COLS` trong `common.py` nếu muốn giống hệt sách) |
| `hdf5_getters` (module phụ của MSD) | đọc trực tiếp bằng `h5py` | không phải tải thêm file, đọc được cả file summary gộp |
| `euclid_dist` (cosine + chênh lệch độ lớn) | mặc định Euclid trên không gian chuẩn hoá, giữ công thức sách ở `--metric book` | công thức của sách cộng thêm `cosine²` nên hai bài **cùng hướng** lại bị coi là xa nhau — giữ lại để đối chiếu |
| Lặp qua toàn bộ bài hát để demo | nhận `--song-id` / `--title` từ dòng lệnh | đúng mô tả "user provides a song-ID as input" |

## 7. Gợi ý mở rộng

* Chọn `k` bằng phương pháp *elbow*: chạy bước 3 với `--k` từ 2→20 rồi vẽ WSSSE.
* Thêm đặc trưng: `duration`, `time_signature`, hoặc trung bình các `segments_timbre`.
* So sánh với **collaborative filtering** (`pyspark.ml.recommendation.ALS`) trên
  Taste Profile subset của MSD — phần sách nêu ở đầu mục 11.3.
* Đánh giá chất lượng: kiểm tra tỉ lệ bài gợi ý cùng nghệ sĩ / cùng thể loại (tag Last.fm).
* Viết bằng API hiện đại `pyspark.ml` (DataFrame + `VectorAssembler` + `StandardScaler`
  + `KMeans`) thay cho `pyspark.mllib` (RDD, đang ở chế độ maintenance).

## 8. Tài liệu tham khảo

1. A. Bahga, V. Madisetti. *Big Data Science & Analytics: A Hands-On Approach*, mục 11.3.
2. T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, P. Lamere. *The Million Song Dataset*, ISMIR 2011.
