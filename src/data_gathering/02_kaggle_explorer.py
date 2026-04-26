"""
02_kaggle_explorer.py
=====================
Explores the Kaggle API to discover NIRF ranking, placement, and
India colleges datasets. Also searches for news headline corpora
for NLP baseline calibration.

Requires: KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import os
import sys
import json
import time
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Environment variable keys
ENV_USERNAME = "KAGGLE_USERNAME"
ENV_KEY = "KAGGLE_KEY"  # mapped from KAGGLE_API_TOKEN in env file

# Searches to run (label -> query string)
SEARCH_QUERIES = {
    "NIRF Rankings": "NIRF ranking",
    "NIRF Placement": "NIRF placement India",
    "India Colleges": "India colleges cities",
    "India Higher Education": "India higher education",
    "College Placement India": "college placement India salary",
    "India News Headlines": "India news headlines",
    "Million Headlines": "million news headlines",
    "Education Policy India": "education policy India NEP",
}

# Known high-value dataset slugs to check directly
TARGET_SLUGS = [
    "nagarshivam/india-colleges-and-cities",
    "atharvmohanjadhav/nirf-2024-college-rankings-dataset",
    "therohk/million-headlines",
]

# Columns we care about (from the research CSV)
TARGET_COLUMNS = [
    "placement", "salary", "lpa", "median", "placed",
    "approved_intake", "intake", "graduation", "graduated",
    "nirf_rank", "rank", "score", "tlr", "rpc", "go",
    "state", "city", "college", "university", "institution",
    "fees", "fee", "ug", "pg",
    "karnataka", "tamil", "nadu",
    "autonomous", "enrollment", "enrolment",
    "category", "sc", "st", "obc", "gender", "female", "male",
    "year", "academic",
]

# Output paths
METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "metadata")
OUTPUT_FILE = os.path.join(METADATA_DIR, "kaggle_catalog.json")


# ──────────────────────────────────────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestResults:
    """Collects and reports test results for each step."""
    def __init__(self):
        self.tests = []
    
    def add(self, name, passed, detail=""):
        self.tests.append({"name": name, "passed": passed, "detail": detail})
        icon = "✓" if passed else "✗"
        print(f"  TEST {icon}  {name}: {detail}")
    
    def summary(self):
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t["passed"])
        failed = total - passed
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY: {passed}/{total} passed, {failed} failed")
        if failed > 0:
            print("Failed tests:")
            for t in self.tests:
                if not t["passed"]:
                    print(f"  ✗ {t['name']}: {t['detail']}")
        print(f"{'='*70}\n")
        return {"total": total, "passed": passed, "failed": failed, "tests": self.tests}


def check_column_relevance(columns):
    """Check how many target column keywords appear in the column list."""
    if not columns:
        return []
    col_text = " ".join(columns).lower()
    return [kw for kw in TARGET_COLUMNS if kw in col_text]


# ──────────────────────────────────────────────────────────────────────────────
# Main Exploration Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  KAGGLE API — Dataset Discovery")
    print(f"  Target: NIRF placement, India colleges, news headlines")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    tests = TestResults()
    catalog = {
        "timestamp": datetime.now().isoformat(),
        "searches": {},
        "target_slug_details": {},
        "promising_datasets": [],
    }
    
    # ── Step 1: Check credentials ─────────────────────────────────────────
    print("\n[STEP 1] Checking Kaggle credentials...")
    
    username = os.environ.get(ENV_USERNAME)
    # Support both KAGGLE_KEY and KAGGLE_API_TOKEN from env file
    key = os.environ.get(ENV_KEY) or os.environ.get("KAGGLE_API_TOKEN")
    
    # The kaggle library expects KAGGLE_KEY, so propagate it
    if key and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = key
    
    if not username or not key:
        msg = (
            f"Missing environment variables.\n"
            f"  Set: export {ENV_USERNAME}=your_username\n"
            f"  Set: export {ENV_KEY}=your_key\n"
            f"  Or place kaggle.json in ~/.kaggle/"
        )
        tests.add("Credentials", False, msg)
        print(f"  ✗ {msg}")
        tests.summary()
        return
    
    tests.add("Credentials", True, f"Username: {username}")
    
    # ── Step 2: Authenticate ──────────────────────────────────────────────
    print("\n[STEP 2] Authenticating with Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        tests.add("Authentication", True, "Successfully authenticated")
    except Exception as e:
        tests.add("Authentication", False, f"Auth failed: {e}")
        print(f"  ✗ Kaggle authentication failed: {e}")
        tests.summary()
        return
    
    # ── Step 3: Search datasets ───────────────────────────────────────────
    print(f"\n[STEP 3] Searching across {len(SEARCH_QUERIES)} queries...")
    
    all_datasets = {}  # Deduplicated by slug
    
    for label, query in SEARCH_QUERIES.items():
        print(f"\n  → Searching: '{query}' (category: {label})")
        try:
            datasets = api.dataset_list(search=query, sort_by="votes")
            # Convert to list (it's a generator/paginated)
            ds_list = list(datasets)[:10]  # Limit to top 10
            
            print(f"    Found {len(ds_list)} results")
            
            search_record = {
                "query": query,
                "results_count": len(ds_list),
                "datasets": []
            }
            
            for ds in ds_list:
                slug = str(ds.ref)
                title = str(ds.title) if hasattr(ds, 'title') else slug
                size = str(ds.size) if hasattr(ds, 'size') else "?"
                last_updated = str(ds.lastUpdated) if hasattr(ds, 'lastUpdated') else "?"
                usability = str(ds.usabilityRating) if hasattr(ds, 'usabilityRating') else "?"
                
                ds_record = {
                    "slug": slug,
                    "title": title,
                    "size": size,
                    "last_updated": last_updated,
                    "usability_rating": usability,
                }
                search_record["datasets"].append(ds_record)
                
                if slug not in all_datasets:
                    all_datasets[slug] = ds_record
                
                print(f"    • {title}")
                print(f"      Slug: {slug} | Size: {size} | Updated: {last_updated}")
            
            catalog["searches"][label] = search_record
            tests.add(
                f"Search: {label}",
                len(ds_list) > 0,
                f"{len(ds_list)} datasets found"
            )
            
        except Exception as e:
            tests.add(f"Search: {label}", False, f"Error: {e}")
            print(f"    ✗ Search failed: {e}")
        
        time.sleep(0.5)
    
    # ── Step 4: Check known target slugs ──────────────────────────────────
    print(f"\n[STEP 4] Checking {len(TARGET_SLUGS)} known target datasets...")
    
    for slug in TARGET_SLUGS:
        print(f"\n  → Checking: {slug}")
        try:
            # Get dataset file list without downloading
            files = api.dataset_list_files(slug)
            file_list = files.files if hasattr(files, 'files') else files
            
            file_records = []
            for f in file_list:
                fname = str(f.name) if hasattr(f, 'name') else str(f)
                fsize = str(f.size) if hasattr(f, 'size') else "?"
                file_records.append({"name": fname, "size": fsize})
                print(f"    File: {fname} ({fsize})")
            
            catalog["target_slug_details"][slug] = {
                "slug": slug,
                "files": file_records,
                "file_count": len(file_records),
            }
            
            tests.add(
                f"Target slug: {slug.split('/')[-1][:30]}",
                len(file_records) > 0,
                f"{len(file_records)} files found"
            )
            
        except Exception as e:
            tests.add(
                f"Target slug: {slug.split('/')[-1][:30]}",
                False,
                f"Error: {e}"
            )
            print(f"    ✗ Failed: {e}")
        
        time.sleep(0.5)
    
    # ── Step 5: Download & peek at CSVs from target slugs ─────────────────
    print(f"\n[STEP 5] Downloading & peeking at target dataset columns...")
    
    import tempfile
    import pandas as pd
    
    for slug in TARGET_SLUGS:
        print(f"\n  → Downloading: {slug}")
        try:
            # Download to a temp directory
            temp_dir = os.path.join(METADATA_DIR, "..", "temp_kaggle")
            os.makedirs(temp_dir, exist_ok=True)
            
            api.dataset_download_files(slug, path=temp_dir, unzip=True, quiet=True)
            
            # Find and peek at CSV files
            csv_files = []
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(".csv"):
                        csv_files.append(os.path.join(root, f))
            
            for csv_path in csv_files[:3]:  # Limit to 3 CSVs per dataset
                fname = os.path.basename(csv_path)
                print(f"\n    CSV: {fname}")
                try:
                    df = pd.read_csv(csv_path, nrows=5, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, nrows=5, encoding="latin-1")
                
                columns = list(df.columns)
                relevance = check_column_relevance(columns)
                
                print(f"      Columns ({len(columns)}): {columns[:15]}{'...' if len(columns) > 15 else ''}")
                print(f"      Shape (first 5 rows): {df.shape}")
                print(f"      Relevant keywords: {relevance[:10]}")
                
                # Check for state column and Karnataka/TN data
                state_cols = [c for c in columns if any(
                    kw in c.lower() for kw in ["state", "location", "city", "region"]
                )]
                if state_cols:
                    for sc in state_cols:
                        # Read more rows to check for state values
                        try:
                            df_full = pd.read_csv(csv_path, usecols=[sc], encoding="utf-8")
                        except UnicodeDecodeError:
                            df_full = pd.read_csv(csv_path, usecols=[sc], encoding="latin-1")
                        unique_vals = df_full[sc].dropna().unique()
                        state_matches = [
                            v for v in unique_vals 
                            if any(s.lower() in str(v).lower() for s in ["karnataka", "tamil nadu", "tamil"])
                        ]
                        if state_matches:
                            print(f"      ★ State column '{sc}' contains: {state_matches}")
                        else:
                            print(f"      State column '{sc}' unique values ({len(unique_vals)}): {list(unique_vals)[:10]}")
                
                # Sample data
                print(f"      Sample row:\n{df.head(1).to_string()}")
                
                catalog["promising_datasets"].append({
                    "slug": slug,
                    "csv_file": fname,
                    "columns": columns,
                    "relevance_keywords": relevance,
                    "shape_sample": list(df.shape),
                    "state_columns": state_cols,
                })
                
                tests.add(
                    f"CSV peek: {fname[:30]}",
                    len(columns) > 0 and len(relevance) > 0,
                    f"{len(columns)} cols, {len(relevance)} relevant matches"
                )
            
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            tests.add(f"Download: {slug.split('/')[-1][:30]}", False, f"Error: {e}")
            print(f"    ✗ Download/peek failed: {e}")
        
        time.sleep(1)
    
    # ── Step 6: Save metadata ─────────────────────────────────────────────
    print(f"\n[STEP 6] Saving metadata catalog...")
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    
    file_exists = os.path.exists(OUTPUT_FILE)
    file_size = os.path.getsize(OUTPUT_FILE) if file_exists else 0
    tests.add(
        "Metadata saved",
        file_exists and file_size > 100,
        f"Saved to {OUTPUT_FILE} ({file_size:,} bytes)"
    )
    
    # ── Final Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KAGGLE DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"  Total unique datasets found: {len(all_datasets)}")
    print(f"  Target slugs checked: {len(TARGET_SLUGS)}")
    print(f"  CSVs analyzed: {len(catalog['promising_datasets'])}")
    
    test_summary = tests.summary()
    catalog["test_results"] = test_summary
    
    # Re-save with test results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    
    return catalog


if __name__ == "__main__":
    main()
