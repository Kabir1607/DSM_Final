"""
04_news_api_explorer.py
=======================
Explores MediaStack and NewsAPI for regional Indian news coverage
about NEP, higher education, and institutional friction in
Karnataka and Tamil Nadu.

Requires environment variables:
  MEDIASTACK_KEY  — Free tier: 500 requests/month
  NEWSAPI_KEY     — Free tier: 100 requests/day, 1-month lookback
"""

import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MEDIASTACK_BASE = "http://api.mediastack.com/v1/news"
NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# Targeted search queries for both APIs
# We keep these minimal to conserve rate-limited free tiers
SEARCH_KEYWORDS = {
    "NEP Education Policy": "NEP education policy Karnataka",
    "Higher Education Karnataka": "higher education Karnataka",
    "Higher Education Tamil Nadu": "higher education Tamil Nadu",
    "Seat Blocking KEA": "seat matrix KEA Karnataka",
    "UGC Education Reform": "UGC education reform India",
    "University Autonomy India": "university autonomy India",
    "Vocational Skills Education": "vocational education skill India",
    "NEP Implementation": "NEP implementation 2020",
}

# Regional domains to filter on (for NewsAPI)
REGIONAL_DOMAINS = [
    "thehindu.com",
    "deccanherald.com",
    "newindianexpress.com",
    "ndtv.com",
    "hindustantimes.com",
]

# Output paths
METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "metadata")
OUTPUT_FILE = os.path.join(METADATA_DIR, "news_api_catalog.json")


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
# MediaStack Functions
# ──────────────────────────────────────────────────────────────────────────────

def mediastack_search(api_key, keywords, country="in", limit=5):
    """Search MediaStack for news articles."""
    params = {
        "access_key": api_key,
        "keywords": keywords,
        "countries": country,
        "limit": limit,
        "sort": "published_desc",
        "languages": "en",
    }
    try:
        resp = requests.get(MEDIASTACK_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {resp.status_code}: {e}"}
    except Exception as e:
        return {"error": str(e)}


def parse_mediastack_response(data):
    """Parse MediaStack response into standardized format."""
    if "error" in data:
        return {"count": 0, "error": data["error"], "articles": []}
    
    articles_raw = data.get("data", [])
    pagination = data.get("pagination", {})
    total = pagination.get("total", 0)
    
    articles = []
    for art in articles_raw:
        articles.append({
            "title": art.get("title", ""),
            "description": (art.get("description", "") or "")[:200],
            "url": art.get("url", ""),
            "source": art.get("source", ""),
            "published_at": art.get("published_at", ""),
            "category": art.get("category", ""),
            "language": art.get("language", ""),
            "country": art.get("country", ""),
        })
    
    return {
        "count": total,
        "returned": len(articles),
        "articles": articles,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NewsAPI Functions
# ──────────────────────────────────────────────────────────────────────────────

def newsapi_search(api_key, query, domains=None, page_size=5,
                   from_date=None, to_date=None, sort_by="relevancy"):
    """Search NewsAPI /v2/everything endpoint."""
    params = {
        "apiKey": api_key,
        "q": query,
        "pageSize": page_size,
        "sortBy": sort_by,
        "language": "en",
    }
    if domains:
        params["domains"] = ",".join(domains)
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    
    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {resp.status_code}: {e}", "status_code": resp.status_code}
    except Exception as e:
        return {"error": str(e)}


def parse_newsapi_response(data):
    """Parse NewsAPI response into standardized format."""
    if "error" in data:
        return {"count": 0, "error": data["error"], "articles": []}
    
    if data.get("status") != "ok":
        return {"count": 0, "error": data.get("message", "Unknown error"), "articles": []}
    
    total = data.get("totalResults", 0)
    articles_raw = data.get("articles", [])
    
    articles = []
    for art in articles_raw:
        source = art.get("source", {})
        articles.append({
            "title": art.get("title", ""),
            "description": (art.get("description", "") or "")[:200],
            "url": art.get("url", ""),
            "source_name": source.get("name", ""),
            "source_id": source.get("id", ""),
            "published_at": art.get("publishedAt", ""),
        })
    
    return {
        "count": total,
        "returned": len(articles),
        "articles": articles,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Exploration Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MediaStack + NewsAPI — Regional News Discovery")
    print(f"  Target: NEP, Higher Education coverage from Indian regional outlets")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    tests = TestResults()
    catalog = {
        "timestamp": datetime.now().isoformat(),
        "mediastack": {"searches": {}, "available_fields": []},
        "newsapi": {"searches": {}, "available_fields": []},
    }
    
    # ── Step 1: Check credentials ─────────────────────────────────────────
    print("\n[STEP 1] Checking API credentials...")
    
    mediastack_key = os.environ.get("MEDIASTACK_KEY") or os.environ.get("MEDIASTACK_API_KEY")
    newsapi_key = os.environ.get("NEWSAPI_KEY") or os.environ.get("newsapi_key")
    
    ms_available = bool(mediastack_key)
    na_available = bool(newsapi_key)
    
    tests.add("MediaStack key", ms_available, 
              f"Key found ({mediastack_key[:8]}...)" if ms_available else "MISSING — set MEDIASTACK_KEY")
    tests.add("NewsAPI key", na_available,
              f"Key found ({newsapi_key[:8]}...)" if na_available else "MISSING — set NEWSAPI_KEY")
    
    if not ms_available and not na_available:
        print("  ✗ No API keys available. Skipping news API exploration.")
        tests.summary()
        return
    
    # ── Step 2: MediaStack exploration ────────────────────────────────────
    if ms_available:
        print(f"\n[STEP 2a] MediaStack — Testing {len(SEARCH_KEYWORDS)} queries...")
        print("  ⚠ Note: Free tier limited to 500 requests/month. Using conservative limits.")
        
        ms_request_count = 0
        
        for label, keywords in SEARCH_KEYWORDS.items():
            if ms_request_count >= 5:  # Limit to 5 queries to conserve free tier
                print(f"\n  ⚠ Stopping MediaStack at {ms_request_count} queries to conserve rate limit")
                break
            
            print(f"\n  → MediaStack: '{keywords}' (category: {label})")
            data = mediastack_search(mediastack_key, keywords, limit=5)
            parsed = parse_mediastack_response(data)
            ms_request_count += 1
            
            if parsed.get("error"):
                tests.add(f"MS: {label}", False, f"Error: {parsed['error']}")
                catalog["mediastack"]["searches"][label] = {"error": str(parsed["error"])}
            else:
                print(f"    Total results: {parsed['count']}")
                print(f"    Returned: {parsed['returned']}")
                
                if parsed["articles"]:
                    # Record available fields
                    if not catalog["mediastack"]["available_fields"]:
                        catalog["mediastack"]["available_fields"] = list(parsed["articles"][0].keys())
                    
                    print(f"    Sample articles:")
                    for i, art in enumerate(parsed["articles"][:3]):
                        print(f"      {i+1}. {art['title'][:70]}")
                        print(f"         Source: {art['source']} | Date: {art['published_at'][:10] if art['published_at'] else '?'}")
                
                catalog["mediastack"]["searches"][label] = {
                    "keywords": keywords,
                    "total_results": parsed["count"],
                    "returned": parsed["returned"],
                    "articles": parsed["articles"],
                }
                
                tests.add(
                    f"MS: {label}",
                    parsed["count"] > 0,
                    f"{parsed['count']} total, {parsed['returned']} returned"
                )
            
            time.sleep(1)  # Respect rate limits
        
        catalog["mediastack"]["requests_used"] = ms_request_count
    else:
        print("\n[STEP 2a] MediaStack — SKIPPED (no API key)")
    
    # ── Step 3: NewsAPI exploration ───────────────────────────────────────
    if na_available:
        print(f"\n[STEP 2b] NewsAPI — Testing queries with domain filtering...")
        print("  ⚠ Note: Free tier limited to 100 requests/day, 1-month lookback.")
        
        na_request_count = 0
        
        # Calculate date range (free tier = last month)
        today = datetime.now()
        one_month_ago = (today - timedelta(days=28)).strftime("%Y-%m-%d")
        
        # Test 1: Broad NEP search
        for label, query in list(SEARCH_KEYWORDS.items())[:5]:  # Limit queries
            if na_request_count >= 5:
                print(f"\n  ⚠ Stopping NewsAPI at {na_request_count} queries to conserve rate limit")
                break
            
            print(f"\n  → NewsAPI: '{query}' (category: {label})")
            data = newsapi_search(
                newsapi_key, query,
                from_date=one_month_ago,
                page_size=5
            )
            parsed = parse_newsapi_response(data)
            na_request_count += 1
            
            if parsed.get("error"):
                tests.add(f"NA: {label}", False, f"Error: {parsed['error']}")
                catalog["newsapi"]["searches"][label] = {"error": str(parsed["error"])}
            else:
                print(f"    Total results: {parsed['count']}")
                print(f"    Returned: {parsed['returned']}")
                
                if parsed["articles"]:
                    if not catalog["newsapi"]["available_fields"]:
                        catalog["newsapi"]["available_fields"] = list(parsed["articles"][0].keys())
                    
                    print(f"    Sample articles:")
                    for i, art in enumerate(parsed["articles"][:3]):
                        print(f"      {i+1}. {art['title'][:70]}")
                        print(f"         Source: {art['source_name']} | Date: {art['published_at'][:10] if art['published_at'] else '?'}")
                
                catalog["newsapi"]["searches"][label] = {
                    "query": query,
                    "total_results": parsed["count"],
                    "returned": parsed["returned"],
                    "articles": parsed["articles"],
                }
                
                tests.add(
                    f"NA: {label}",
                    parsed["count"] > 0,
                    f"{parsed['count']} total, {parsed['returned']} returned"
                )
            
            time.sleep(1)
        
        # Test 2: Domain-filtered search
        if na_request_count < 8:
            print(f"\n  → NewsAPI with domain filter: {', '.join(REGIONAL_DOMAINS[:3])}")
            data = newsapi_search(
                newsapi_key,
                "education policy Karnataka",
                domains=REGIONAL_DOMAINS[:3],
                from_date=one_month_ago,
                page_size=5
            )
            parsed = parse_newsapi_response(data)
            na_request_count += 1
            
            if parsed.get("error"):
                tests.add("NA: Domain-filtered", False, f"Error: {parsed['error']}")
            else:
                print(f"    Total results (domain-filtered): {parsed['count']}")
                for art in parsed["articles"][:3]:
                    print(f"      • [{art['source_name']}] {art['title'][:60]}")
                
                catalog["newsapi"]["searches"]["Domain-filtered Karnataka"] = {
                    "query": "education policy Karnataka",
                    "domains": REGIONAL_DOMAINS[:3],
                    "total_results": parsed["count"],
                    "articles": parsed["articles"],
                }
                tests.add(
                    "NA: Domain-filtered",
                    parsed["count"] >= 0,
                    f"{parsed['count']} results from regional domains"
                )
        
        catalog["newsapi"]["requests_used"] = na_request_count
    else:
        print("\n[STEP 2b] NewsAPI — SKIPPED (no API key)")
    
    # ── Step 4: Compare field schemas ─────────────────────────────────────
    print(f"\n[STEP 3] Comparing API field schemas for NLP pipeline compatibility...")
    
    ms_fields = catalog["mediastack"]["available_fields"]
    na_fields = catalog["newsapi"]["available_fields"]
    
    print(f"  MediaStack fields: {ms_fields}")
    print(f"  NewsAPI fields: {na_fields}")
    
    # Check for the critical fields needed for RoBERTa pipeline
    required_fields = ["title", "description", "url", "published_at"]
    
    if ms_fields:
        ms_has_required = [f for f in required_fields if any(f in mf.lower() for mf in ms_fields)]
        tests.add(
            "MS schema compatibility",
            len(ms_has_required) >= 3,
            f"Has {len(ms_has_required)}/4 required fields: {ms_has_required}"
        )
    
    if na_fields:
        na_has_required = [f for f in required_fields if any(f in nf.lower() for nf in na_fields)]
        tests.add(
            "NA schema compatibility",
            len(na_has_required) >= 3,
            f"Has {len(na_has_required)}/4 required fields: {na_has_required}"
        )
    
    # ── Step 5: Save metadata ─────────────────────────────────────────────
    print(f"\n[STEP 4] Saving metadata catalog...")
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
    print("  NEWS API DISCOVERY SUMMARY")
    print("=" * 70)
    
    if ms_available:
        ms_searches = catalog["mediastack"]["searches"]
        ms_total = sum(s.get("total_results", 0) for s in ms_searches.values() if isinstance(s, dict))
        print(f"\n  MediaStack:")
        print(f"    Queries run: {catalog['mediastack'].get('requests_used', 0)}")
        print(f"    Total results across queries: {ms_total}")
        print(f"    Available fields: {ms_fields}")
    
    if na_available:
        na_searches = catalog["newsapi"]["searches"]
        na_total = sum(s.get("total_results", 0) for s in na_searches.values() if isinstance(s, dict))
        print(f"\n  NewsAPI:")
        print(f"    Queries run: {catalog['newsapi'].get('requests_used', 0)}")
        print(f"    Total results across queries: {na_total}")
        print(f"    Available fields: {na_fields}")
    
    test_summary = tests.summary()
    catalog["test_results"] = test_summary
    
    # Re-save with test results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    
    return catalog


if __name__ == "__main__":
    main()
