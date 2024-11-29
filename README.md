# LLM-blocking

## Dolma Preprocessing Pipeline

This repository provides tools for managing Dolma JSON files and preparing data for further analysis. The pipeline includes functionalities for loading, preprocessing, filtering, and frequency analysis.

---

### 1. JSON Request

- Use one of the data_load files to collect JSON files
    1. **Get a single JSON file by overwriting:** `data_load.py`
    2. **Get multiple JSON files without overwriting**:`data_load_1123.py`
        - For large JSON files, save it by splitting into smaller parts: `data_load_1123_split.py`

### 2. Preprocessing

- Steps for `json_preprocessing.py`
    1. **Integration**
        - Consolidates all JSON files into one CSV file.
        - Extracts key elements such as words and sentences.
    2. **Simple Word Filtering**
        - Keeps only English words with length > 3.
        - Excludes words present in: NLTK word list, WordNet, Unimorph
    3. **Arch Filtering**
    4. **Spell Correction Filtering**
    5. **Levenshtein Filtering**
- Output
    - Filtered and removed CSV files for each processing stage in `./data/csv/`.
    - Final output saved in `./data/words/`.

### 3. Frequency Analysis

- Analyze word frequencies in the final processed dataset.
    1. within the collected raw JSON files.
    2. within the entire DOLMA dataset.

---

## Directory Structure
    ```
    .
    ├── data/
    │   ├── json/
    │   ├── csv/
    │   └── words/
    ├── data_load.py
    ├── data_load_1123.py
    ├── data_load_1123_split.py
    ├── json_preprocessing.py
    └── frequency.py

    ```