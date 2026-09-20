# Case Study: Song Recommendation System (Spark)

This is a class practice project for the *Big Data Analytics Algorithms* course.
The codebase is inspired by Section 11.3 of Bahga & Madisetti, *Big Data Science & Analytics: A Hands-On Approach* (Box 11.4 → Box 11.7), updated for **Python 3 + Spark 3.x**.

The system utilizes **content-based filtering**: it does not require user listening history or ratings. Instead, it relies solely on the audio features of songs extracted from the Million Song Dataset.

## 1. Flow

```text
   File .h5 (Million Song Subset)
              │  Step 1 (Box 11.4)
              ▼
   data/songs.csv   ──────────────► Step 2 (Box 11.5): Top 10 songs/artists per year
              │  Step 3 (Box 11.6)
              ▼
   K-Means (k = 10) on 4 features
              │
              ├─► data/kmeans_model.p     (Cluster centers + normalization params)
              └─► data/songs_clustered/   (Dataset + `cluster` column)
                          │  Step 4 (Box 11.7)
                          ▼
     Input song → Find cluster → Calculate distance with songs in the SAME cluster
                → Sort ascendingly → Top 10 similar songs
```

**Core Idea**: Pre-cluster the data to **narrow down the search space** (only compare within 1/k of the dataset), allowing the recommendation process to run in near real-time.

## 2. Folder Structure

| File | Role | Book Reference |
|---|---|---|
| [src/common.py](src/common.py) | CSV schema, features, distance functions, Spark utilities | — |
| [src/step1_extract_h5_to_csv.py](src/step1_extract_h5_to_csv.py) | Extract HDF5 → CSV metadata | Box 11.4 |
| [src/step2_top_songs_artists.py](src/step2_top_songs_artists.py) | Find top 10 songs / artists per year | Box 11.5 |
| [src/step3_cluster_songs.py](src/step3_cluster_songs.py) | K-Means clustering using Spark MLlib | Box 11.6 |
| [src/step4_recommend_songs.py](src/step4_recommend_songs.py) | Recommend similar songs | Box 11.7 |
| [src/tools/make_demo_csv.py](src/tools/make_demo_csv.py) | Generate mock data for testing without the full dataset | — |
| [src/tools/selftest_no_spark.py](src/tools/selftest_no_spark.py) | Re-run logic for steps 3–4 in pure Python (no Spark needed) | — |

**CSV Format** (Generated in Step 1, includes header):
```csv
artist_id, artist_name, song_id, song_name, loudness, song_hotttnesss,
tempo, key_confidence, mode_confidence, year, artist_hotttnesss, duration
```
The first 10 columns maintain the order from the book; the last 2 columns were added for artist ranking and display purposes.

## 3. Environment Setup (Docker)

This project assumes a running Spark Docker container is available in the environment.

1. Ensure the Spark container is started.
2. Access the Spark container's shell and navigate to this project:
   ```bash
   docker exec -it <spark_container_name> bash
   cd <project_directory_name>
   pip install -r requirements.txt
   ```
*(Note: Ensure the project directory is volume-mapped into the container so that the code and data are automatically synced.)*

## 4. Execution Instruction

Download the dataset. We use a **subset of 10,000 songs** (≈2 GB) as required by the assignment:

```bash
mkdir -p data && cd data
# Option 1: If using macOS/Linux host
curl -O http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz
# Option 2: Alternative if curl is not found (e.g., inside some Docker containers)
wget http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz

tar -xzf millionsongsubset.tar.gz          # Extracts to data/MillionSongSubset/
cd ..
```
*(Note: Because the workspace is synced, this download step can be executed directly on the host machine's terminal instead of inside the Docker container.)*

Official site: <http://millionsongdataset.com/>. If the downloaded version contains the aggregated `subset_msd_summary_file.h5`, pointing `--input` directly to it will significantly speed up Step 1.

## 5. Execute Pipeline

Run the following commands inside the Spark container (or the local environment if dependencies are installed):

```bash
# Step 1 — HDF5 to CSV (Add --limit 500 for a quick test)
python3 src/step1_extract_h5_to_csv.py --input data/MillionSongSubset --output output_evidence/songs.csv

# Step 2 — Top 10 songs / artists by year
python3 src/step2_top_songs_artists.py --input output_evidence/songs.csv \
    --from-year 1990 --to-year 1999 --output-json output_evidence/top_by_year.json

# Step 3 — K-Means Clustering (k = 10)
python3 src/step3_cluster_songs.py --input output_evidence/songs.csv --k 10 --overwrite

# Step 4 — Recommend 10 similar songs
python3 src/step4_recommend_songs.py --song-id SOICLQB12A8C13637C --top 10
python3 src/step4_recommend_songs.py --title "Exodus" --output-json output_evidence/recommend.json
python3 src/step4_recommend_songs.py --random --metric book   # Use the book's specific formula
```

If running on a real Spark cluster (YARN), submit the job like this:
```bash
spark-submit --master yarn --py-files src/common.py src/step3_cluster_songs.py --input hdfs:///songs.csv
```

## 6. Verification

To quickly test the logic without downloading the 2GB dataset, mock data can be generated:

```bash
# 1. Generate demo dataset
python3 src/tools/make_demo_csv.py --output data/songs_demo.csv --n 2000

# 2. Verify logic using pure Python (No Spark/Java required)
python3 src/tools/selftest_no_spark.py --input data/songs_demo.csv

# 3. Test Spark clustering with the demo data
python3 src/step3_cluster_songs.py --input data/songs_demo.csv --overwrite

# 4. Test recommendations
python3 src/step4_recommend_songs.py --random
```

## 7. Technical Note

The following modifications were made compared to the original code in the textbook to modernize the implementation:

| Book (2016, Python 2 / Spark 1.x) | This Repository | Rationale |
|---|---|---|
| `print x`, `line.encode('utf-8')` | Python 3 syntax | The book was written for Python 2. |
| `KMeans.train(..., runs=100)` | Removed `runs` parameter | This parameter was deprecated and removed in Spark 2.0. |
| `line.split(",")` | Used `csv.reader` | Song/Artist names containing commas caused column misalignment. |
| Clustered on raw values | Z-score normalization before clustering | Features like loudness (dB), tempo (BPM), and confidence (0–1) are on vastly different scales. Without normalization, tempo dominates the distance metric. |
| Features: loudness, song_hotttnesss, tempo, key_confidence | loudness, tempo, key_confidence, mode_confidence | `song_hotttnesss` has NaN values in many subset songs (Modify `FEATURE_COLS` in `common.py` to match the book exactly if desired). |
| `hdf5_getters` (MSD sub-module) | Read directly via `h5py` | Eliminates the need to download extra files and supports reading the aggregated summary file. |
| `euclid_dist` (cosine + magnitude diff) | Default Euclidean on normalized space; kept book's formula under `--metric book` | The book's formula adds `cosine²`, causing two songs in the **same direction** to be considered far apart. Retained strictly for comparison. |
| Iterated over all songs for demo | Accepts `--song-id` / `--title` from CLI | Aligns with the textbook description: "user provides a song-ID as input". |

## 8. AI Tools

- **Google Gemini 3.1 Pro**: Used for code review, documentation structuring, translation, and generating the Docker Compose configuration to ensure a professional and reproducible setup.

## 9. Acknowledgements

1. A. Bahga, V. Madisetti. *Big Data Science & Analytics: A Hands-On Approach*, Section 11.3.
2. T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, P. Lamere. *The Million Song Dataset*, ISMIR 2011.
