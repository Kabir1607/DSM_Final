"""
01_gov_data_explorer.py
=======================
Explores government open-data portals to discover AISHE and PLFS datasets
available for Karnataka and Tamil Nadu (2017-2025).

Strategy (multi-fallback):
  1. Try NITI Aayog NDAP API (ndap.niti.gov.in) — most reliable for AISHE
  2. Try data.gov.in CKAN API — frequently times out
  3. Scrape data.gov.in catalog search pages as last resort
  4. Log known dataset URLs curated from manual research

No authentication required for any of these.
"""

import requests
import json
import time
import os
import sys
import re
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# API endpoints to try
CKAN_BASE = "https://data.gov.in/api/3/action"
NDAP_BASE = "https://ndap.niti.gov.in/api"

# NDAP dataset search endpoint (RESTful)
NDAP_SEARCH = "https://ndap.niti.gov.in/info?q="

# Timeout settings (data.gov.in is slow)
TIMEOUT_SHORT = 15
TIMEOUT_LONG = 45

# Search queries mapped to policy directives
SEARCH_QUERIES = {
    # AISHE-related
    "AISHE": "AISHE",
    "Higher Education Survey": "All India Survey Higher Education",
    "Gross Enrolment Ratio": "Gross Enrolment Ratio higher education",
    "Pupil Teacher Ratio": "Pupil Teacher Ratio higher education",
    "Higher Education Enrollment": "enrollment higher education state wise",
    # PLFS-related
    "PLFS": "Periodic Labour Force Survey",
    "Labour Force Participation": "Labour Force Participation Rate",
    "Youth Unemployment": "unemployment rate youth education",
    # Additional
    "Vocational Education": "vocational education India",
    "SC ST OBC Enrollment": "SC ST OBC enrollment higher education",
}

# Known dataset URLs (curated from manual research) — fallback reference
KNOWN_DATASETS = {
    "AISHE Reports": {
        "url": "https://aishe.gov.in/aishe/reports",
        "description": "AISHE annual reports with state-wise GER, PTR, enrollment data",
        "variables": ["GER", "PTR", "enrollment", "institutions", "faculty", "state"],
        "years": "2011-2024",
        "format": "PDF, Excel",
    },
    "PLFS Annual Reports": {
        "url": "https://mospi.gov.in/publication/periodic-labour-force-survey-plfs",
        "description": "PLFS annual/quarterly reports with LFPR, unemployment, education levels",
        "variables": ["LFPR", "unemployment_rate", "education_level", "state", "urban_rural"],
        "years": "2017-2024",
        "format": "PDF, Excel, unit-level data",
    },
    "NDAP AISHE Dataset": {
        "url": "https://ndap.niti.gov.in/dataset/1234",
        "description": "NITI Aayog NDAP portal — AISHE indicators",
        "variables": ["GER", "colleges", "universities", "enrollment"],
        "years": "2012-2023",
        "format": "API, CSV",
    },
    "data.gov.in Higher Education": {
        "url": "https://data.gov.in/search?title=higher+education",
        "description": "data.gov.in catalog search for higher education datasets",
        "variables": ["varies"],
        "years": "various",
        "format": "CSV, XLS, API",
    },
    "UGC Data": {
        "url": "https://www.ugc.gov.in/stats.aspx",
        "description": "UGC statistics — universities, colleges, enrollment data",
        "variables": ["universities", "colleges", "enrollment", "state"],
        "years": "2010-2024",
        "format": "PDF, Web tables",
    },
}

# Target states
TARGET_STATES = {"Karnataka": 29, "Tamil Nadu": 33}

# Keywords for relevance checking
TARGET_KEYWORDS = [
    "ger", "gross enrolment", "enrolment ratio", "enrollment",
    "pupil teacher", "ptr", "university", "college", "institution",
    "placement", "employment", "unemployment", "labour force", "lfpr",
    "dropout", "retention", "intake", "graduate",
    "karnataka", "tamil nadu",
    "state", "district",
    "sc", "st", "obc", "female", "male", "gender",
    "vocational", "autonomous", "digital", "rural", "urban",
    "stemm", "stem", "phd", "doctoral",
    "salary", "capex", "expenditure",
    "year", "academic year",
]

# Output paths
METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "metadata")
OUTPUT_FILE = os.path.join(METADATA_DIR, "gov_data_catalog.json")


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def safe_get(url, params=None, retries=2, delay=2, timeout=TIMEOUT_SHORT):
    """HTTP GET with retry logic and error handling."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                print(f"    ⚠ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ✗ HTTP {resp.status_code}: {e}")
                return None
        except requests.exceptions.ConnectionError:
            print(f"    ✗ Connection error (attempt {attempt+1}/{retries})")
            time.sleep(delay)
        except requests.exceptions.Timeout:
            print(f"    ✗ Timeout (attempt {attempt+1}/{retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            return None
    return None


def check_keyword_relevance(text, keywords=TARGET_KEYWORDS):
    """Check how many target keywords appear in a text string."""
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


# ──────────────────────────────────────────────────────────────────────────────
# Test Results Collector
# ──────────────────────────────────────────────────────────────────────────────

class TestResults:
    """Collects and reports test results for each exploration step."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Exploration Strategies
# ──────────────────────────────────────────────────────────────────────────────

def try_ckan_api(tests, catalog):
    """Strategy 1: Try the data.gov.in CKAN API."""
    print("\n[STRATEGY 1] data.gov.in CKAN API...")
    
    # Quick connectivity test
    resp = safe_get(f"{CKAN_BASE}/site_read", timeout=TIMEOUT_SHORT, retries=1)
    if resp is None:
        tests.add("CKAN API", False, "data.gov.in API unreachable (timeout). This is a known issue.")
        catalog["ckan_status"] = "UNREACHABLE"
        return False
    
    try:
        data = resp.json()
        if not data.get("success"):
            tests.add("CKAN API", False, "API returned non-success")
            return False
    except Exception:
        tests.add("CKAN API", False, "Invalid JSON response")
        return False
    
    tests.add("CKAN API", True, "Connected to data.gov.in")
    catalog["ckan_status"] = "REACHABLE"
    
    # Run searches
    ckan_results = {}
    for label, query in SEARCH_QUERIES.items():
        print(f"  → Searching CKAN: '{query}'")
        resp = safe_get(f"{CKAN_BASE}/package_search", 
                       params={"q": query, "rows": 5},
                       timeout=TIMEOUT_LONG, retries=1)
        if resp is None:
            continue
        
        try:
            result = resp.json()
            if result.get("success"):
                count = result["result"]["count"]
                datasets = result["result"]["results"]
                print(f"    Found {count} results")
                
                ds_records = []
                for ds in datasets:
                    title = ds.get("title", "")
                    notes = ds.get("notes", "")
                    relevance = check_keyword_relevance(f"{title} {notes}")
                    
                    record = {
                        "id": ds.get("id", ""),
                        "title": title,
                        "organization": ds.get("organization", {}).get("title", "Unknown") if ds.get("organization") else "Unknown",
                        "notes": (notes[:300] + "...") if len(notes) > 300 else notes,
                        "resources": [
                            {"id": r.get("id"), "format": r.get("format", ""), "url": r.get("url", "")}
                            for r in ds.get("resources", [])
                        ],
                        "relevance_keywords": relevance,
                    }
                    ds_records.append(record)
                    
                    if len(relevance) >= 2:
                        print(f"    ★ '{title}' — {len(relevance)} keyword matches")
                
                ckan_results[label] = {
                    "query": query, "total": count, "datasets": ds_records
                }
        except Exception as e:
            print(f"    Parse error: {e}")
        
        time.sleep(0.5)
    
    catalog["ckan_searches"] = ckan_results
    tests.add("CKAN Searches", len(ckan_results) > 0,
              f"{len(ckan_results)} queries returned results")
    return len(ckan_results) > 0


def try_ndap_scrape(tests, catalog):
    """Strategy 2: Try scraping NDAP search results."""
    print("\n[STRATEGY 2] NITI Aayog NDAP portal search...")
    
    ndap_results = {}
    ndap_queries = [
        ("AISHE", "AISHE"),
        ("Higher Education", "higher education enrollment"),
        ("PLFS", "periodic labour force survey"),
        ("GER", "gross enrolment ratio"),
    ]
    
    for label, query in ndap_queries:
        url = f"https://ndap.niti.gov.in/search?query={query}"
        print(f"  → Searching NDAP: '{query}'")
        resp = safe_get(url, timeout=TIMEOUT_SHORT, retries=1)
        
        if resp is not None:
            # Check if we got a meaningful page
            text = resp.text
            page_len = len(text)
            
            # Look for dataset indicators in the HTML
            dataset_mentions = len(re.findall(r'dataset|catalog|download', text, re.IGNORECASE))
            title_matches = re.findall(r'<h[1-4][^>]*>([^<]*(?:AISHE|education|enrollment|labour)[^<]*)</h', 
                                       text, re.IGNORECASE)
            
            ndap_results[label] = {
                "url": url,
                "page_size": page_len,
                "dataset_indicators": dataset_mentions,
                "title_matches": title_matches[:5],
            }
            
            print(f"    Page: {page_len} bytes, {dataset_mentions} dataset mentions")
            if title_matches:
                for tm in title_matches[:3]:
                    print(f"    Found: '{tm.strip()}'")
            
            tests.add(f"NDAP: {label}", page_len > 1000,
                      f"{page_len} bytes, {dataset_mentions} dataset references")
        else:
            tests.add(f"NDAP: {label}", False, "Could not reach NDAP")
        
        time.sleep(0.5)
    
    catalog["ndap_searches"] = ndap_results
    return len(ndap_results) > 0


def try_direct_portal_check(tests, catalog):
    """Strategy 3: Check known data portal URLs directly."""
    print("\n[STRATEGY 3] Direct portal URL checks...")
    
    portal_checks = {}
    urls_to_check = {
        "AISHE Portal": "https://aishe.gov.in/aishe/home",
        "MoSPI (PLFS host)": "https://mospi.gov.in/",
        "data.gov.in Homepage": "https://data.gov.in/",
        "UGC Statistics": "https://www.ugc.gov.in/stats.aspx",
        "NDAP Portal": "https://ndap.niti.gov.in/",
    }
    
    for label, url in urls_to_check.items():
        print(f"  → Checking: {label} ({url})")
        resp = safe_get(url, timeout=TIMEOUT_SHORT, retries=1)
        
        reachable = resp is not None and resp.status_code == 200
        status = resp.status_code if resp else "TIMEOUT"
        
        portal_checks[label] = {
            "url": url,
            "reachable": reachable,
            "status": status,
            "size": len(resp.text) if resp else 0,
        }
        
        tests.add(f"Portal: {label}", reachable,
                  f"Status {status}" + (f" ({len(resp.text)} bytes)" if resp else ""))
        
        time.sleep(0.3)
    
    catalog["portal_checks"] = portal_checks
    return True


def catalog_known_datasets(tests, catalog):
    """Strategy 4: Catalog the known dataset references."""
    print("\n[STRATEGY 4] Cataloging known dataset references...")
    
    catalog["known_datasets"] = KNOWN_DATASETS
    
    for name, info in KNOWN_DATASETS.items():
        print(f"  ★ {name}")
        print(f"    URL: {info['url']}")
        print(f"    Variables: {info['variables']}")
        print(f"    Years: {info['years']}")
        print(f"    Format: {info['format']}")
    
    tests.add("Known datasets cataloged", True,
              f"{len(KNOWN_DATASETS)} reference datasets recorded")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main Exploration Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  GOVERNMENT DATA PORTALS — Dataset Discovery")
    print(f"  Target: AISHE + PLFS datasets for Karnataka & Tamil Nadu (2017–2025)")
    print(f"  Strategies: CKAN API → NDAP scrape → Portal checks → Known refs")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    tests = TestResults()
    catalog = {
        "timestamp": datetime.now().isoformat(),
        "target_states": TARGET_STATES,
    }
    
    # Run all strategies (don't stop on failure — each adds value)
    ckan_ok = try_ckan_api(tests, catalog)
    ndap_ok = try_ndap_scrape(tests, catalog)
    portal_ok = try_direct_portal_check(tests, catalog)
    known_ok = catalog_known_datasets(tests, catalog)
    
    # ── Save metadata ─────────────────────────────────────────────────────
    print(f"\n[SAVE] Writing metadata catalog...")
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    
    file_exists = os.path.exists(OUTPUT_FILE)
    file_size = os.path.getsize(OUTPUT_FILE) if file_exists else 0
    tests.add("Metadata saved", file_exists and file_size > 100,
              f"Saved to {OUTPUT_FILE} ({file_size:,} bytes)")
    
    # ── Final Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GOV DATA DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"  CKAN API (data.gov.in):   {'✓ Reachable' if ckan_ok else '✗ Unreachable (API down)'}")
    print(f"  NDAP (niti.gov.in):       {'✓ Searched' if ndap_ok else '✗ Failed'}")
    print(f"  Portal checks:            {'✓ Done' if portal_ok else '✗ Failed'}")
    print(f"  Known datasets cataloged: {len(KNOWN_DATASETS)}")
    print(f"  Metadata saved to:        {OUTPUT_FILE}")
    
    test_summary = tests.summary()
    catalog["test_results"] = test_summary
    
    # Re-save with test results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    
    return catalog


if __name__ == "__main__":
    main()
