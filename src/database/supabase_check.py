"""
supabase_check.py
=================
A script to verify the Supabase PostgreSQL connection and ensure 
all data has been successfully ingested into the cloud database.
"""

import os
import urllib.parse
from sqlalchemy import create_engine, text

# 1. Database Connection Details
# It is highly recommended to set this as an environment variable:
# export SUPABASE_URL="postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres"
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", 
    "postgresql://postgres.jsnczuqnlewxtdpcuurl:Iamanidiot%231607%23Unique%402026@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres"
)

def main():
    print("=" * 70)
    print("  Supabase Database Verification Script")
    print("=" * 70)
    
    try:
        # Initialize engine with sslmode=require for Supabase
        engine = create_engine(SUPABASE_URL, connect_args={'sslmode': 'require'})
        
        # Test basic connection
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
            print(f"✅ Connection Successful!\nServer Version: {version}\n")
    except Exception as e:
        print("❌ Connection Failed. Please check your SUPABASE_URL.")
        print(f"Error: {e}")
        return

    # 2. Check Table Counts
    print("[1] Row Counts per Table")
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
            except Exception:
                print(f" {table:25} : ❌ ERROR (Table might not exist)")

    # 3. Data Integrity Checks
    print("\n[2] Data Integrity Checks")
    print("-" * 50)
    
    with engine.connect() as conn:
        # Check pgvector extension (Crucial for your embeddings)
        try:
            vector_ext = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")).fetchone()
            if vector_ext:
                print(f" ✅ pgvector extension is installed (Version: {vector_ext[0]})")
            else:
                print(" ❌ pgvector extension is MISSING!")
        except Exception:
            pass

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