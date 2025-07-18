# Data Preprocessing Report

This report summarizes the process of cleaning a raw GPS Parquet dataset to resolve schema inconsistencies, data corruption, and performance bottlenecks.

---

### 1. Initial Data Challenges

The source dataset consisted of 7 Parquet files containing 1.16 billion rows. Direct analysis was unfeasible due to several issues:

*   **Schema Inconsistency**: `column03` (timestamp) and `column09` (angle) had mixed data types (`DATETIME`/`STRING` and `NUMBER`/`STRING`) across files.
*   **Data Corruption**: The dataset contained invalid timestamps defaulting to `1970-01-01` and GPS coordinates outside of standard ranges.
*   **Performance Bottlenecks**: Queries on the raw data led to out-of-memory errors and timeouts.

---

### 2. Processing Methodology

A Python script (`clean_parquet_files.py`) was used to process the data with a DuckDB engine configured for large datasets (`memory_limit='12GB'`, `threads=8`, `preserve_insertion_order=false`).

The process for each file was as follows:
1.  **Read Data**: Ingested using `read_parquet` with `union_by_name=true` to handle schema differences.
2.  **Standardize Types**: `TRY_CAST` was applied to safely convert columns to their target types (`TIMESTAMP`, `DOUBLE`, `BIGINT`), turning invalid entries into `NULL`.
3.  **Write Cleaned File**: A new Parquet file was written using `Snappy` compression and a `ROW_GROUP_SIZE` of 50,000 for balanced performance.
4.  **Verification**: Post-cleaning, a global view over all processed files confirmed schema consistency across all 1.16 billion rows.

---

### 3. Key Findings from Cleaned Data

Analysis of the standardized dataset revealed the following:

| Metric | Finding |
| :--- | :--- |
| **Temporal Focus** | Data is concentrated in late Nov/early Dec 2019. |
| **Geographic Center** | Operations are centered around (115.6° E, 39.8° N). |
| **Vehicle State** | Median speed of 0.0 suggests a high volume of stationary/idle time. |
| **Dominant Companies** | ZHTC and JYJ account for over 60% of all records. |
| **Invalid Data** | Over 275,000 invalid GPS coordinate pairs were successfully filtered. |

The dataset is now validated, schema-consistent, and ready for advanced analysis. 