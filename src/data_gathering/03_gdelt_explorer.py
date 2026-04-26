"""
03_gdelt_explorer.py
====================
Explores the GDELT DOC 2.0 API to discover news articles about NEP,
higher education, and institutional friction in Karnataka and Tamil Nadu.

No authentication required. Free and open API.

API Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
"""

import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta
from urllib.parse import quote

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# India FIPS code for sourcecountry filter
INDIA_FIPS = "IN"

# Exploration queries organized by category
# Each query tests a different aspect of the project's data needs
EXPLORATION_QUERIES = {
    # ── Core NEP Policy Coverage ──
    "NEP General (India)": {
        "query": '("National Education Policy" OR "NEP 2020") sourcecountry:india',
        "mode": "artlist",
        "purpose": "Broad NEP coverage across all Indian sources",
    },
    "NEP Karnataka": {
        "query": '("National Education Policy" OR "NEP") Karnataka sourcecountry:india',
        "mode": "artlist",
        "purpose": "NEP coverage specific to Karnataka (treatment group)",
    },
    "NEP Tamil Nadu": {
        "query": '("National Education Policy" OR "NEP") "Tamil Nadu" sourcecountry:india',
        "mode": "artlist",
        "purpose": "NEP coverage specific to Tamil Nadu (control group)",
    },
    
    # ── Higher Education Broad Coverage ──
    "Higher Education Karnataka": {
        "query": '("higher education" OR university OR college) Karnataka sourcecountry:india',
        "mode": "artlist",
        "purpose": "General HE coverage for Karnataka",
    },
    "Higher Education Tamil Nadu": {
        "query": '("higher education" OR university OR college) "Tamil Nadu" sourcecountry:india',
        "mode": "artlist",
        "purpose": "General HE coverage for Tamil Nadu",
    },
    
    # ── Institutional Friction Points (from research literature) ──
    "Seat Blocking Crisis": {
        "query": '("seat matrix" OR "seat blocking" OR "KEA" OR "vacant seats") Karnataka sourcecountry:india',
        "mode": "artlist",
        "purpose": "Administrative efficiency friction: KEA seat-blocking crisis",
    },
    "UGC IKS Mandates": {
        "query": '("UGC" OR "Indian Knowledge Systems" OR "IKS") ("education" OR "university") sourcecountry:india',
        "mode": "artlist",
        "purpose": "Coverage of UGC IKS mandates causing institutional friction",
    },
    "Faculty Protests Education": {
        "query": '("faculty protests" OR "teacher workload" OR "teacher shortage") education Karnataka sourcecountry:india',
        "mode": "artlist",
        "purpose": "Faculty friction and workload complaints",
    },
    "University Autonomy": {
        "query": '("university autonomy" OR "autonomous college" OR "institutional autonomy") India sourcecountry:india',
        "mode": "artlist",
        "purpose": "Autonomy & restructuring policy directive tracking",
    },
    "Education Vocational Skills": {
        "query": '("vocational education" OR "skill development" OR "multidisciplinary") Karnataka sourcecountry:india',
        "mode": "artlist",
        "purpose": "Vocational/technical skill integration coverage",
    },
    "Digital Education India": {
        "query": '("digital education" OR "smart classroom" OR "digital divide") India education sourcecountry:india',
        "mode": "artlist",
        "purpose": "Digital divide & infrastructure coverage",
    },
    
    # ── Domain-Specific Searches ──
    "Deccan Herald NEP": {
        "query": '("NEP" OR "education policy") domain:deccanherald.com',
        "mode": "artlist",
        "purpose": "Karnataka regional newspaper coverage of NEP",
    },
    "The Hindu Education Policy": {
        "query": '("education policy" OR "NEP" OR "higher education") domain:thehindu.com',
        "mode": "artlist",
        "purpose": "Tamil Nadu regional newspaper coverage",
    },
    
    # ── Tone/Sentiment Timeline Queries ──
    "NEP Tone Timeline": {
        "query": '("National Education Policy" OR "NEP 2020") sourcecountry:india',
        "mode": "timelinetone",
        "purpose": "Sentiment trend of NEP coverage over time",
    },
    "NEP Volume Timeline": {
        "query": '("National Education Policy" OR "NEP 2020") sourcecountry:india',
        "mode": "timelinevol",
        "purpose": "Volume of NEP coverage over time",
    },
}

# Output paths  
METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "metadata")
OUTPUT_FILE = os.path.join(METADATA_DIR, "gdelt_catalog.json")


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


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def gdelt_query(query_str, mode="artlist", fmt="json", maxrecords=25,
                timespan=None, startdatetime=None, enddatetime=None):
    """Execute a GDELT DOC 2.0 API query."""
    params = {
        "query": query_str,
        "mode": mode,
        "format": fmt,
        "maxrecords": maxrecords,
    }
    if timespan:
        params["timespan"] = timespan
    if startdatetime:
        params["STARTDATETIME"] = startdatetime
    if enddatetime:
        params["ENDDATETIME"] = enddatetime
    
    try:
        resp = requests.get(GDELT_BASE, params=params, timeout=30)
        resp.raise_for_status()
        
        # GDELT sometimes returns empty response or HTML error pages
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type or fmt == "json":
            try:
                return resp.json()
            except json.JSONDecodeError:
                # GDELT may return empty body for no-result queries
                return {"articles": [], "note": "Empty or non-JSON response"}
        elif "csv" in content_type or fmt == "csv":
            return {"raw_csv": resp.text[:2000], "note": "CSV response"}
        else:
            return {"raw_text": resp.text[:2000], "note": f"Non-JSON content-type: {content_type}"}
    
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {resp.status_code}: {str(e)}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def extract_article_info(data):
    """Extract article count, sample titles, and tone from GDELT response."""
    articles = data.get("articles", [])
    if not articles:
        return {"count": 0, "samples": [], "avg_tone": None}
    
    samples = []
    tones = []
    for art in articles[:10]:
        title = art.get("title", "No title")
        url = art.get("url", "")
        tone = art.get("tone", 0)
        seendate = art.get("seendate", "")
        domain = art.get("domain", "")
        language = art.get("language", "")
        source_country = art.get("sourcecountry", "")
        
        samples.append({
            "title": title,
            "url": url[:100],
            "tone": tone,
            "date": seendate,
            "domain": domain,
            "language": language,
            "source_country": source_country,
        })
        if isinstance(tone, (int, float)):
            tones.append(tone)
    
    avg_tone = sum(tones) / len(tones) if tones else None
    
    return {
        "count": len(articles),
        "samples": samples,
        "avg_tone": round(avg_tone, 2) if avg_tone is not None else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Exploration Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  GDELT DOC 2.0 API — News Sentiment Discovery")
    print(f"  Target: NEP, Higher Education, Institutional Friction")
    print(f"  Geographic: Karnataka (treatment) vs Tamil Nadu (control)")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    tests = TestResults()
    catalog = {
        "timestamp": datetime.now().isoformat(),
        "api_base": GDELT_BASE,
        "queries": {},
        "temporal_tests": {},
        "summary_stats": {},
    }
    
    # ── Step 1: Test API connectivity ─────────────────────────────────────
    print("\n[STEP 1] Testing GDELT API connectivity...")
    test_data = gdelt_query("test", mode="artlist", maxrecords=1)
    api_works = "error" not in test_data
    tests.add(
        "API Connectivity",
        api_works,
        "GDELT reachable" if api_works else f"Error: {test_data.get('error', 'Unknown')}"
    )
    
    if not api_works:
        print("  ✗ Cannot reach GDELT API")
        tests.summary()
        return
    
    # ── Step 2: Run all exploration queries ───────────────────────────────
    print(f"\n[STEP 2] Running {len(EXPLORATION_QUERIES)} exploration queries...")
    
    for label, config in EXPLORATION_QUERIES.items():
        query = config["query"]
        mode = config["mode"]
        purpose = config["purpose"]
        
        print(f"\n  → Query: {label}")
        print(f"    Purpose: {purpose}")
        print(f"    GDELT query: {query}")
        print(f"    Mode: {mode}")
        
        data = gdelt_query(query, mode=mode, maxrecords=25)
        
        if "error" in data:
            tests.add(f"Query: {label}", False, f"Error: {data['error']}")
            catalog["queries"][label] = {"error": data["error"]}
            time.sleep(1)
            continue
        
        if mode == "artlist":
            info = extract_article_info(data)
            print(f"    Articles found: {info['count']}")
            print(f"    Average tone: {info['avg_tone']}")
            
            if info["samples"]:
                print(f"    Sample articles:")
                for i, s in enumerate(info["samples"][:5]):
                    print(f"      {i+1}. [{s['tone']:.1f}] {s['title'][:80]}")
                    print(f"         Source: {s['domain']} | Date: {s['date'][:10] if s['date'] else '?'}")
            
            catalog["queries"][label] = {
                "query": query,
                "mode": mode,
                "purpose": purpose,
                "article_count": info["count"],
                "avg_tone": info["avg_tone"],
                "samples": info["samples"],
            }
            
            tests.add(
                f"Query: {label}",
                info["count"] > 0,
                f"{info['count']} articles, avg tone: {info['avg_tone']}"
            )
        
        elif mode in ["timelinetone", "timelinevol"]:
            # Timeline data has a different structure
            timeline = data.get("timeline", [])
            if timeline:
                series = timeline[0].get("data", []) if timeline else []
                print(f"    Timeline data points: {len(series)}")
                if series:
                    dates = [s.get("date", "") for s in series[:3]]
                    values = [s.get("value", 0) for s in series[:3]]
                    print(f"    First 3 data points: dates={dates}, values={values}")
                
                catalog["queries"][label] = {
                    "query": query,
                    "mode": mode,
                    "purpose": purpose,
                    "data_points": len(series),
                    "sample_points": series[:5],
                }
                tests.add(
                    f"Query: {label}",
                    len(series) > 0,
                    f"{len(series)} timeline data points"
                )
            else:
                catalog["queries"][label] = {
                    "query": query, "mode": mode,
                    "raw_keys": list(data.keys()),
                    "note": "Timeline structure varies"
                }
                tests.add(
                    f"Query: {label}",
                    len(data) > 1,  # More than just empty
                    f"Response keys: {list(data.keys())[:5]}"
                )
        
        time.sleep(1.5)  # Be polite — GDELT is free
    
    # ── Step 3: Test temporal range capabilities ──────────────────────────
    print(f"\n[STEP 3] Testing temporal range capabilities...")
    
    # GDELT DOC 2.0 officially searches last 3 months
    # Test with explicit STARTDATETIME/ENDDATETIME
    now = datetime.utcnow()
    
    temporal_tests = {
        "Last 7 days": {
            "timespan": "7days",
        },
        "Last 30 days": {
            "timespan": "30days",
        },
        "Last 3 months": {
            "timespan": "3months",
        },
        "Specific recent window": {
            "startdatetime": (now - timedelta(days=60)).strftime("%Y%m%d%H%M%S"),
            "enddatetime": (now - timedelta(days=30)).strftime("%Y%m%d%H%M%S"),
        },
    }
    
    nep_query = '("National Education Policy" OR "NEP") sourcecountry:india'
    
    for label, params in temporal_tests.items():
        print(f"\n  → Temporal test: {label}")
        data = gdelt_query(
            nep_query, mode="artlist", maxrecords=10,
            timespan=params.get("timespan"),
            startdatetime=params.get("startdatetime"),
            enddatetime=params.get("enddatetime"),
        )
        
        if "error" in data:
            tests.add(f"Temporal: {label}", False, f"Error: {data['error']}")
        else:
            info = extract_article_info(data)
            print(f"    Articles: {info['count']}, Avg tone: {info['avg_tone']}")
            catalog["temporal_tests"][label] = {
                "params": params,
                "article_count": info["count"],
                "avg_tone": info["avg_tone"],
            }
            tests.add(
                f"Temporal: {label}",
                info["count"] >= 0,  # 0 is valid for narrow windows
                f"{info['count']} articles"
            )
        
        time.sleep(1.5)
    
    # ── Step 4: CSV format test for bulk extraction later ─────────────────
    print(f"\n[STEP 4] Testing CSV output format for future bulk extraction...")
    csv_data = gdelt_query(
        '("National Education Policy") sourcecountry:india',
        mode="artlist", fmt="csv", maxrecords=5
    )
    
    if csv_data and "raw_csv" in csv_data:
        csv_preview = csv_data["raw_csv"][:500]
        print(f"    CSV preview:\n{csv_preview}")
        catalog["csv_format_test"] = csv_preview
        tests.add("CSV format", len(csv_preview) > 10, f"{len(csv_preview)} chars")
    elif csv_data and "error" not in csv_data:
        # Might be direct text
        tests.add("CSV format", True, "Response received (may be non-CSV)")
    else:
        tests.add("CSV format", False, "No CSV data returned")
    
    # ── Step 5: Save metadata ─────────────────────────────────────────────
    print(f"\n[STEP 5] Saving metadata catalog...")
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
    print("  GDELT DISCOVERY SUMMARY")
    print("=" * 70)
    
    # Compute summary stats
    article_counts = {}
    tone_scores = {}
    for label, qdata in catalog["queries"].items():
        if "article_count" in qdata:
            article_counts[label] = qdata["article_count"]
        if "avg_tone" in qdata and qdata["avg_tone"] is not None:
            tone_scores[label] = qdata["avg_tone"]
    
    print(f"\n  Article counts by query:")
    for label, count in sorted(article_counts.items(), key=lambda x: -x[1]):
        tone = tone_scores.get(label, "N/A")
        print(f"    {count:4d} articles | tone: {tone:>6} | {label}")
    
    catalog["summary_stats"] = {
        "total_queries": len(EXPLORATION_QUERIES),
        "queries_with_results": sum(1 for c in article_counts.values() if c > 0),
        "article_counts": article_counts,
        "tone_scores": tone_scores,
    }
    
    test_summary = tests.summary()
    catalog["test_results"] = test_summary
    
    # Re-save with test results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    
    return catalog


if __name__ == "__main__":
    main()
