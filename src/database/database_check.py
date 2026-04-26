"""
database_check.py
=================
A script to verify that all data has been successfully ingested into the PostgreSQL database.
It counts the rows in all tables and performs basic data integrity checks.
"""

import os
import urllib.parse
from sqlalchemy import create_engine, text

# Database Connection Details
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

def main():
    engine = create_engine(DB_URL)
    
    print("=" * 70)
    print("  NEP Database Verification Script")
    print("=" * 70)
    
    # 1. Check Table Counts
    print("\n[1] Row Counts per Table")
    print("-" * 50)
    tables = [
        "state_reference",
        "institutions",
        "nirf_rankings",
        "placements",
        "macro_controls",
        "aishe_infrastructure",
        "aishe_enrollment",
        "news_corpus",
        "policy_documents"
    ]
    
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f" {table:25} : {result:>12,} rows")
            except Exception as e:
                print(f" {table:25} : ERROR (Table might not exist)")

    # 2. Basic Integrity Checks
    print("\n[2] Data Integrity Checks")
    print("-" * 50)
    
    with engine.connect() as conn:
        # Check years in NIRF
        try:
            min_year, max_year = conn.execute(text("SELECT MIN(year), MAX(year) FROM nirf_rankings")).fetchone()
            print(f" NIRF Rankings Years       : {min_year} to {max_year}")
        except Exception: 
            pass
            
        # Check Macro Controls
        try:
            min_date, max_date = conn.execute(text("SELECT MIN(date), MAX(date) FROM macro_controls")).fetchone()
            print(f" Macro Controls Date Range : {min_date} to {max_date}")
        except Exception: 
            pass

        # Check source distribution in news
        try:
            print("\n News Corpus Sources:")
            sources = conn.execute(text("SELECT source_name, COUNT(*) FROM news_corpus GROUP BY source_name ORDER BY count DESC")).fetchall()
            if not sources:
                print("  (No news data found)")
            for source, count in sources:
                print(f"  - {source:21} : {count:>12,} articles")
        except Exception: 
            pass
            
        # Check AISHE Coverage
        try:
            print("\n AISHE Enrollment Years:")
            aishe_years = conn.execute(text("SELECT survey_year, COUNT(*) FROM aishe_enrollment GROUP BY survey_year ORDER BY survey_year")).fetchall()
            if not aishe_years:
                print("  (No enrollment data found)")
            for yr, count in aishe_years:
                print(f"  - {yr:21} : {count:>12,} records")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(" Verification Complete.")

if __name__ == "__main__":
    main()
