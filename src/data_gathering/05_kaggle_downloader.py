"""
05_kaggle_downloader.py
=======================
Downloads all priority Kaggle datasets for the NEP policy analysis,
organizes them into categorized folders, and creates detailed metadata
files for each dataset.

Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json)

Target datasets:
  - NIRF Rankings (multi-year 2016-2025 + 2024 detailed)
  - India Colleges & Cities (placement, fees, state data)
  - AISHE Higher Education Analytics
  - Indian Engineering College Placements
  - India News Headlines (for sentiment analysis)
  - Indian Financial News (2003-2020)
  - Economic Times Headlines (2022-2025)
  - Unemployment in India
"""

import os
import sys
import json
import time
import shutil
import hashlib
from datetime import datetime

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Datasets to download, organized by category
# Each entry: slug -> {category, description, relevance}
DATASETS = {
    # ── NIRF Rankings & Placement ──
    "iitanshravan/nirf-rankings-dataset-20162025": {
        "category": "nirf",
        "folder": "nirf_rankings_2016_2025",
        "description": "NIRF Rankings from 2016 to 2025 — multi-year dataset ideal for DiD time-series analysis",
        "relevance": "Primary treatment/control outcome variable (institutional rankings over time)",
        "variables_expected": ["rank", "score", "state", "year", "institution"],
    },
    "atharvmohanjadhav/nirf-2024-college-rankings-dataset": {
        "category": "nirf",
        "folder": "nirf_2024_detailed",
        "description": "NIRF 2024 College Rankings with TLR, RPC, GO, OI, Perception sub-scores",
        "relevance": "Detailed sub-score breakdown for 2024 — cross-sectional validation",
        "variables_expected": ["TLR", "RPC", "GO", "OI", "Score", "Rank", "State", "Field"],
    },
    "satyajeet69/top-engineering-colleges-in-india-2025": {
        "category": "nirf",
        "folder": "top_engineering_2025",
        "description": "Top Engineering Colleges in India 2025 with placement data",
        "relevance": "Latest placement outcomes for engineering institutions",
        "variables_expected": ["college", "placement", "salary", "state"],
    },

    # ── India Colleges & Institutions ──
    "nagarshivam/india-colleges-and-cities": {
        "category": "institutions",
        "folder": "india_colleges_cities",
        "description": "India Colleges, Schools, and Cities dataset with fees, placement, and ratings",
        "relevance": "Institutional-level placement_avg_lpa and fees data with state column — key for DiD",
        "variables_expected": ["name", "city", "state", "fees_ug_inr", "placement_avg_lpa", "nirf_rank"],
    },

    # ── Placement Data ──
    "vishardmehta/indian-engineering-college-placement-dataset": {
        "category": "placements",
        "folder": "engineering_placements",
        "description": "Indian Engineering College Placement Dataset",
        "relevance": "Direct placement outcome data for engineering colleges",
        "variables_expected": ["placement", "salary", "college", "state"],
    },
    "siddheshshivdikar/college-placement": {
        "category": "placements",
        "folder": "college_placement_general",
        "description": "College Placement Dataset — general placement outcomes",
        "relevance": "Supplementary placement data",
        "variables_expected": ["placement", "salary", "degree"],
    },

    # ── AISHE / Higher Education ──
    "rajanand/aishe": {
        "category": "aishe",
        "folder": "aishe_higher_education",
        "description": "Higher Education Analytics — AISHE data on Kaggle",
        "relevance": "Core AISHE metrics: GER, enrollment, institutions — bypasses broken gov API",
        "variables_expected": ["GER", "enrollment", "state", "year", "universities"],
    },

    # ── Employment / Unemployment ──
    "piyushborhade/unemployment-in-india": {
        "category": "employment",
        "folder": "unemployment_india",
        "description": "Unemployment in India dataset",
        "relevance": "State-level unemployment data — control variable for DiD (Xist)",
        "variables_expected": ["unemployment_rate", "state", "date"],
    },

    # ── News Headlines (for Sentiment Analysis) ──
    "therohk/india-headlines-news-dataset": {
        "category": "news",
        "folder": "india_news_headlines",
        "description": "India News Headlines Dataset — Indian news corpus",
        "relevance": "Primary corpus for RoBERTa sentiment analysis on education/NEP coverage",
        "variables_expected": ["headline_text", "publish_date"],
    },
    "abhiaero/economic-times-headlines-india-2022-to-2025": {
        "category": "news",
        "folder": "et_headlines_2022_2025",
        "description": "Economic Times Headlines (India) 2022-2025",
        "relevance": "Post-NEP period headlines from major business newspaper",
        "variables_expected": ["headline", "date"],
    },
    "hkapoor/indian-financial-news-articles-20032020": {
        "category": "news",
        "folder": "financial_news_2003_2020",
        "description": "Indian Financial News Articles 2003-2020",
        "relevance": "Pre-NEP and transition period news for sentiment baseline",
        "variables_expected": ["headline", "date", "content"],
    },
}

# Master catalog output
CATALOG_FILE = os.path.join(DATA_DIR, "master_data_catalog.json")


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def file_hash(filepath, algo="md5"):
    """Compute file hash for integrity tracking."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_csv(filepath):
    """Analyze a CSV file and return metadata."""
    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            df = pd.read_csv(filepath, nrows=0, encoding="utf-8")
            df_sample = pd.read_csv(filepath, nrows=5, encoding="utf-8")
            df_shape = pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, nrows=0, encoding="latin-1")
            df_sample = pd.read_csv(filepath, nrows=5, encoding="latin-1")
            df_shape = pd.read_csv(filepath, encoding="latin-1")

        columns = list(df.columns)
        total_rows = len(df_shape)
        dtypes = {col: str(dtype) for col, dtype in df_shape.dtypes.items()}

        # Check for state column with Karnataka/TN
        state_cols = [c for c in columns if any(
            kw in c.lower() for kw in ["state", "location", "region"]
        )]
        state_values = {}
        for sc in state_cols:
            unique = df_shape[sc].dropna().unique().tolist()
            karnataka = [v for v in unique if "karnataka" in str(v).lower()]
            tamil_nadu = [v for v in unique if "tamil" in str(v).lower()]
            state_values[sc] = {
                "total_unique": len(unique),
                "has_karnataka": len(karnataka) > 0,
                "has_tamil_nadu": len(tamil_nadu) > 0,
                "sample_values": [str(v) for v in unique[:15]],
            }

        # Check for year/date columns
        time_cols = [c for c in columns if any(
            kw in c.lower() for kw in ["year", "date", "period", "academic"]
        )]
        time_ranges = {}
        for tc in time_cols:
            vals = df_shape[tc].dropna()
            if len(vals) > 0:
                time_ranges[tc] = {
                    "min": str(vals.min()),
                    "max": str(vals.max()),
                    "unique_count": int(vals.nunique()),
                }

        return {
            "columns": columns,
            "total_rows": total_rows,
            "total_columns": len(columns),
            "dtypes": dtypes,
            "file_size_bytes": os.path.getsize(filepath),
            "state_analysis": state_values,
            "time_analysis": time_ranges,
            "sample_data": df_sample.head(3).to_dict(orient="records"),
            "null_counts": {col: int(v) for col, v in df_shape.isnull().sum().items() if v > 0},
        }
    except Exception as e:
        return {"error": str(e), "file_size_bytes": os.path.getsize(filepath)}


def create_dataset_metadata(slug, config, folder_path, files_info):
    """Create a metadata JSON file for a downloaded dataset."""
    metadata = {
        "dataset_slug": slug,
        "kaggle_url": f"https://www.kaggle.com/datasets/{slug}",
        "category": config["category"],
        "description": config["description"],
        "relevance_to_project": config["relevance"],
        "variables_expected": config["variables_expected"],
        "download_timestamp": datetime.now().isoformat(),
        "local_folder": folder_path,
        "files": files_info,
        "total_files": len(files_info),
        "total_size_bytes": sum(f.get("size_bytes", 0) for f in files_info),
    }

    metadata_path = os.path.join(folder_path, "_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return metadata


# ──────────────────────────────────────────────────────────────────────────────
# Main Download Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  KAGGLE DATASET DOWNLOADER — NEP Policy Analysis")
    print(f"  Downloading {len(DATASETS)} datasets into categorized folders")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Step 1: Authenticate ──────────────────────────────────────────────
    print("\n[STEP 1] Authenticating with Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("  ✓ Authenticated successfully")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        print("  Set KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json in ~/.kaggle/")
        return

    # ── Step 2: Create directory structure ────────────────────────────────
    print("\n[STEP 2] Creating directory structure...")
    categories = set(d["category"] for d in DATASETS.values())
    for cat in categories:
        cat_dir = os.path.join(DATA_DIR, "kaggle", cat)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"  ✓ data/kaggle/{cat}/")

    # ── Step 3: Download each dataset ─────────────────────────────────────
    print(f"\n[STEP 3] Downloading {len(DATASETS)} datasets...")

    master_catalog = {
        "download_timestamp": datetime.now().isoformat(),
        "project": "NEP Policy Analysis — Karnataka DiD",
        "datasets": {},
        "summary": {},
    }

    success_count = 0
    fail_count = 0

    for slug, config in DATASETS.items():
        category = config["category"]
        folder_name = config["folder"]
        folder_path = os.path.join(DATA_DIR, "kaggle", category, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        print(f"\n  {'─'*60}")
        print(f"  → Downloading: {slug}")
        print(f"    Category: {category}")
        print(f"    Target: data/kaggle/{category}/{folder_name}/")

        try:
            # Download and unzip
            api.dataset_download_files(slug, path=folder_path, unzip=True, quiet=False)
            print(f"    ✓ Download complete")

            # Analyze all files in the folder
            files_info = []
            for root, dirs, files in os.walk(folder_path):
                for fname in sorted(files):
                    if fname.startswith("_"):  # Skip our metadata files
                        continue
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, folder_path)
                    fsize = os.path.getsize(fpath)

                    file_record = {
                        "filename": fname,
                        "relative_path": rel_path,
                        "size_bytes": fsize,
                        "size_human": f"{fsize/1024:.1f} KB" if fsize < 1024*1024 else f"{fsize/1024/1024:.1f} MB",
                        "md5": file_hash(fpath),
                    }

                    # Analyze CSVs in detail
                    if fname.endswith(".csv"):
                        print(f"    Analyzing: {fname}...")
                        csv_meta = analyze_csv(fpath)
                        file_record["csv_analysis"] = csv_meta
                        if "error" not in csv_meta:
                            print(f"      Columns ({csv_meta['total_columns']}): {csv_meta['columns'][:8]}{'...' if csv_meta['total_columns'] > 8 else ''}")
                            print(f"      Rows: {csv_meta['total_rows']:,}")
                            if csv_meta['state_analysis']:
                                for sc, sv in csv_meta['state_analysis'].items():
                                    ka = "✓" if sv['has_karnataka'] else "✗"
                                    tn = "✓" if sv['has_tamil_nadu'] else "✗"
                                    print(f"      State '{sc}': Karnataka {ka} | Tamil Nadu {tn} ({sv['total_unique']} states)")
                            if csv_meta['time_analysis']:
                                for tc, tv in csv_meta['time_analysis'].items():
                                    print(f"      Time '{tc}': {tv['min']} → {tv['max']} ({tv['unique_count']} values)")
                        else:
                            print(f"      ⚠ Analysis error: {csv_meta['error']}")
                    else:
                        print(f"    File: {fname} ({file_record['size_human']})")

                    files_info.append(file_record)

            # Create per-dataset metadata
            ds_metadata = create_dataset_metadata(slug, config, folder_path, files_info)
            master_catalog["datasets"][slug] = ds_metadata
            print(f"    ✓ Metadata saved: _metadata.json")

            success_count += 1

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            master_catalog["datasets"][slug] = {
                "dataset_slug": slug,
                "error": str(e),
                "download_timestamp": datetime.now().isoformat(),
            }
            fail_count += 1

        time.sleep(1)  # Brief pause between downloads

    # ── Step 4: Save master catalog ───────────────────────────────────────
    print(f"\n[STEP 4] Saving master catalog...")

    master_catalog["summary"] = {
        "total_datasets": len(DATASETS),
        "successful_downloads": success_count,
        "failed_downloads": fail_count,
        "categories": {cat: sum(1 for d in DATASETS.values() if d["category"] == cat) for cat in categories},
        "download_completed": datetime.now().isoformat(),
    }

    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(master_catalog, f, indent=2, ensure_ascii=False, default=str)

    print(f"  ✓ Master catalog saved: {CATALOG_FILE}")

    # ── Final Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  Successful: {success_count}/{len(DATASETS)}")
    print(f"  Failed:     {fail_count}/{len(DATASETS)}")
    print(f"\n  Folder structure:")

    for cat in sorted(categories):
        cat_datasets = {s: d for s, d in DATASETS.items() if d["category"] == cat}
        print(f"    data/kaggle/{cat}/")
        for slug, config in cat_datasets.items():
            status = "✓" if slug in master_catalog["datasets"] and "error" not in master_catalog["datasets"][slug] else "✗"
            print(f"      {status} {config['folder']}/")

    print(f"\n  Master catalog: {CATALOG_FILE}")
    print("=" * 70)

    return master_catalog


if __name__ == "__main__":
    main()
