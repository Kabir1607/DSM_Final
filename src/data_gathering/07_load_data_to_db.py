"""
07_load_data_to_db.py
=====================
ETL script to load all downloaded data into the PostgreSQL nep_db.
OPTIMIZED FOR BULK LOADING using pandas vectorization and to_sql.
"""

import os
import sys
import json
import urllib.parse
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Add parent to path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database.database_creation import (
    Base, Institution, StateReference, NirfRanking,
    Placement, AisheEnrollment, AisheInfrastructure,
    MacroControl, NewsCorpus
)

# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

# We use fast_executemany for psycopg2 if possible, but method='multi' in to_sql is also fast
engine = create_engine(DB_URL)

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def safe_float(val):
    if pd.isna(val): return None
    try: return float(val)
    except (ValueError, TypeError): return None

def safe_int(val):
    if pd.isna(val): return None
    try:
        f = float(val)
        if np.isinf(f) or np.isnan(f): return None
        return int(f)
    except (ValueError, TypeError, OverflowError): return None

def safe_str(val, max_len=None):
    if pd.isna(val): return None
    s = str(val).strip()
    if max_len: s = s[:max_len]
    return s if s else None

# ──────────────────────────────────────────────────────────────────────────────
# 1. STATE REFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def load_state_reference(session):
    print("\n[1/8] Loading state_reference...")
    aishe_base = os.path.join(DATA_DIR, "kaggle", "aishe", "aishe_higher_education")
    for root, dirs, files in os.walk(aishe_base):
        for f in files:
            if f.endswith("ref_state.csv"):
                df = pd.read_csv(os.path.join(root, f), encoding="latin-1")
                df = df[['st_code', 'name']].drop_duplicates(subset=['st_code'])
                df['name'] = df['name'].str.strip()
                # Bulk insert via to_sql
                df.to_sql('state_reference', con=engine, if_exists='append', index=False)
                print(f"  ✓ Loaded {len(df)} states from {f}")
                return
    print("  ⚠ No ref_state.csv found")

# ──────────────────────────────────────────────────────────────────────────────
# 2. INSTITUTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_institutions(session):
    print("\n[2/8] Loading institutions (Row-by-Row for ID Mapping)...")
    inst_map = {}  # (name_lower, state_lower) -> institution_id

    # 1. NIRF Master
    nirf_path = os.path.join(DATA_DIR, "processed", "NIRF_Master_2016_2025.csv")
    nirf = pd.read_csv(nirf_path)
    count = 0
    for _, row in nirf.iterrows():
        name = safe_str(row.get('Institute Name'), 500)
        state = safe_str(row.get('State'), 100)
        if not name or not state: continue
        key = (name.lower(), state.lower())
        if key not in inst_map:
            inst = Institution(nirf_id=safe_str(row.get('Institute ID'), 50), name=name, city=safe_str(row.get('City'), 100), state=state)
            session.add(inst)
            session.flush()
            inst_map[key] = inst.institution_id
            count += 1
    print(f"  ✓ {count} institutions from NIRF Master")

    # 2. India Colleges
    ic_path = os.path.join(DATA_DIR, "kaggle", "institutions", "india_colleges_cities", "india_colleges.csv")
    ic = pd.read_csv(ic_path)
    count2 = 0
    for _, row in ic.iterrows():
        name = safe_str(row.get('name'), 500)
        state = safe_str(row.get('state'), 100)
        if not name or not state: continue
        key = (name.lower(), state.lower())
        if key not in inst_map:
            inst = Institution(name=name, city=safe_str(row.get('city'), 100), state=state, institution_type=safe_str(row.get('type'), 100))
            session.add(inst)
            session.flush()
            inst_map[key] = inst.institution_id
            count2 += 1
    print(f"  ✓ {count2} new institutions from india_colleges.csv")

    # 3. Top Engineering
    te_path = os.path.join(DATA_DIR, "kaggle", "nirf", "top_engineering_2025", "Untitled spreadsheet - Sheet1.csv")
    te = pd.read_csv(te_path)
    count3 = 0
    for _, row in te.iterrows():
        name = safe_str(row.get('College Name'), 500)
        state = safe_str(row.get('State'), 100)
        if not name or not state: continue
        key = (name.lower(), state.lower())
        if key not in inst_map:
            inst = Institution(name=name, city=safe_str(row.get('Location'), 100), state=state, institution_type=safe_str(row.get('Type'), 100), year_established=safe_int(row.get('Year Established')))
            session.add(inst)
            session.flush()
            inst_map[key] = inst.institution_id
            count3 += 1
    print(f"  ✓ {count3} new institutions from top_engineering_2025")
    return inst_map

# ──────────────────────────────────────────────────────────────────────────────
# 3. NIRF RANKINGS
# ──────────────────────────────────────────────────────────────────────────────

def load_nirf_rankings(session, inst_map):
    print("\n[3/8] Loading nirf_rankings...")
    nirf_path = os.path.join(DATA_DIR, "processed", "NIRF_Master_2016_2025.csv")
    df = pd.read_csv(nirf_path)
    
    rows = []
    for _, row in df.iterrows():
        name = safe_str(row.get('Institute Name'), 500)
        state = safe_str(row.get('State'), 100)
        if not name or not state: continue
        key = (name.lower(), state.lower())
        inst_id = inst_map.get(key)
        if inst_id:
            rows.append({
                'institution_id': inst_id,
                'year': safe_int(row.get('Year')),
                'rank': safe_int(row.get('Rank')),
                'overall_score': safe_float(row.get('Score')),
                'tlr_score': safe_float(row.get('TLR')),
                'rpc_score': safe_float(row.get('RPC')),
                'go_score': safe_float(row.get('GO')),
                'oi_score': safe_float(row.get('OI')),
                'perception_score': safe_float(row.get('PERCEPTION'))
            })
            
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_sql('nirf_rankings', con=engine, if_exists='append', index=False)
        print(f"  ✓ {len(df_out)} NIRF ranking rows loaded via bulk")

# ──────────────────────────────────────────────────────────────────────────────
# 4. PLACEMENTS
# ──────────────────────────────────────────────────────────────────────────────

def load_placements(session, inst_map):
    print("\n[4/8] Loading placements...")
    rows = []
    
    ic_path = os.path.join(DATA_DIR, "kaggle", "institutions", "india_colleges_cities", "india_colleges.csv")
    ic = pd.read_csv(ic_path)
    for _, row in ic.iterrows():
        name, state = safe_str(row.get('name'), 500), safe_str(row.get('state'), 100)
        if not name or not state: continue
        inst_id = inst_map.get((name.lower(), state.lower()))
        rows.append({
            'institution_id': inst_id,
            'avg_salary_lpa': safe_float(row.get('placement_avg_lpa')),
            'fees_ug_inr': safe_float(row.get('fees_ug_inr')),
            'source': 'india_colleges'
        })
        
    te_path = os.path.join(DATA_DIR, "kaggle", "nirf", "top_engineering_2025", "Untitled spreadsheet - Sheet1.csv")
    te = pd.read_csv(te_path)
    for _, row in te.iterrows():
        name, state = safe_str(row.get('College Name'), 500), safe_str(row.get('State'), 100)
        if not name or not state: continue
        inst_id = inst_map.get((name.lower(), state.lower()))
        rows.append({
            'institution_id': inst_id, 'year': 2025,
            'avg_salary_lpa': safe_float(row.get('Average Package (LPA)')),
            'highest_package_lpa': safe_float(row.get('Highest Package (LPA)')),
            'avg_placement_pct': safe_float(row.get('Average Placement %')),
            'fees_ug_inr': safe_float(row.get('Fees (INR/year)')),
            'student_faculty_ratio': safe_float(row.get('Student-Faculty Ratio')),
            'source': 'top_engineering_2025'
        })

    pg_path = os.path.join(DATA_DIR, "kaggle", "placements", "college_placement_general", "detailed_data.csv")
    pg = pd.read_csv(pg_path)
    for _, row in pg.iterrows():
        name = safe_str(row.get('college name'), 500)
        inst_id = None
        if name:
            for k, v in inst_map.items():
                if k[0] == name.lower():
                    inst_id = v
                    break
        salary_raw = safe_float(row.get('Salary'))
        rows.append({
            'institution_id': inst_id, 'year': safe_int(row.get('year')),
            'company_name': safe_str(row.get('name of company'), 255),
            'avg_salary_lpa': salary_raw / 100000.0 if salary_raw else None,
            'source': 'college_placement_general'
        })

    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_sql('placements', con=engine, if_exists='append', index=False)
        print(f"  ✓ {len(df_out)} total placement rows loaded via bulk")

# ──────────────────────────────────────────────────────────────────────────────
# 5. MACRO CONTROLS
# ──────────────────────────────────────────────────────────────────────────────

def load_macro_controls(session):
    print("\n[5/8] Loading macro_controls...")
    unemp_path = os.path.join(DATA_DIR, "kaggle", "employment", "unemployment_india", "Unemployment in India.csv")
    df = pd.read_csv(unemp_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['Region', 'Date'])
    
    df['state'] = df['Region'].str.strip()
    df['area'] = df['Area'].str.strip()
    df['estimated_unemployment_rate'] = pd.to_numeric(df['Estimated Unemployment Rate (%)'], errors='coerce')
    df['estimated_employed'] = pd.to_numeric(df['Estimated Employed'], errors='coerce')
    df['estimated_lfpr'] = pd.to_numeric(df['Estimated Labour Participation Rate (%)'], errors='coerce')
    
    # Vectorized date parsing
    df['date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.date
    df = df.dropna(subset=['date'])
    
    db_cols = ['state', 'date', 'area', 'estimated_unemployment_rate', 'estimated_employed', 'estimated_lfpr']
    df_out = df[db_cols]
    df_out.to_sql('macro_controls', con=engine, if_exists='append', index=False)
    print(f"  ✓ {len(df_out)} macro control rows loaded via bulk")

# ──────────────────────────────────────────────────────────────────────────────
# 6. AISHE INFRASTRUCTURE (BULK FAST)
# ──────────────────────────────────────────────────────────────────────────────

def load_aishe_infrastructure(session):
    print("\n[6/8] Loading aishe_infrastructure (BULK)...")
    aishe_base = os.path.join(DATA_DIR, "kaggle", "aishe", "aishe_higher_education")
    total = 0
    import re

    seen_infra_files = set()
    for dirname in sorted(os.listdir(aishe_base)):
        dirpath = os.path.join(aishe_base, dirname)
        if not os.path.isdir(dirpath): continue
        for f in os.listdir(dirpath):
            if f.endswith("infrastructure.csv") and not f.startswith("course"):
                fpath = os.path.join(dirpath, f)
                real_path = os.path.realpath(fpath)
                if real_path in seen_infra_files: continue
                seen_infra_files.add(real_path)

                m = re.match(r'(\d{4}_\d{2})', f)
                survey_year = m.group(1) if m else f.split('_infrastructure')[0][:10]

                try:
                    df = pd.read_csv(fpath, encoding="latin-1", low_memory=False)
                except Exception: continue

                rename_map = {
                    'id': 'aishe_record_id', 'library': 'has_library', 'laboratory': 'has_laboratory',
                    'computer_center': 'has_computer_center', 'playground': 'has_playground',
                    'solar_power_generation': 'solar_power'
                }
                df = df.rename(columns=rename_map)
                df['survey_year'] = survey_year

                db_cols = ['aishe_record_id', 'has_library', 'has_laboratory', 'has_computer_center', 
                           'has_playground', 'no_of_books', 'no_of_journals', 'no_of_computer_centers', 
                           'no_of_laboratories', 'connectivity_nkn', 'connectivity_nmeict', 
                           'solar_power', 'campus_friendly', 'survey_year']
                
                # Keep only cols that exist
                df_out = df[[c for c in db_cols if c in df.columns]]
                
                # Fast insert
                df_out.to_sql('aishe_infrastructure', con=engine, if_exists='append', index=False, method='multi', chunksize=10000)
                count = len(df_out)
                total += count
                print(f"  ✓ {count} rows from {f}")

    print(f"  → Total infrastructure rows: {total}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. AISHE ENROLLMENT (BULK FAST)
# ──────────────────────────────────────────────────────────────────────────────

def load_aishe_enrollment(session):
    print("\n[7/8] Loading aishe_enrollment (BULK)...")
    aishe_base = os.path.join(DATA_DIR, "kaggle", "aishe", "aishe_higher_education")
    total = 0
    import re

    seen_enroll_files = set()
    for dirname in sorted(os.listdir(aishe_base)):
        dirpath = os.path.join(aishe_base, dirname)
        if not os.path.isdir(dirpath): continue
        for f in os.listdir(dirpath):
            if f.endswith("enrolled_student_count.csv") and not f.startswith("course_") and not f.startswith("enrolled_foreign") and not f.startswith("enrolled_distance"):
                fpath = os.path.join(dirpath, f)
                real_path = os.path.realpath(fpath)
                if real_path in seen_enroll_files: continue
                seen_enroll_files.add(real_path)

                m = re.match(r'(\d{4}_\d{2})', f)
                survey_year = m.group(1) if m else f.split('_enrolled')[0][:10]

                try:
                    df = pd.read_csv(fpath, encoding="latin-1", low_memory=False)
                except Exception: continue

                df = df.rename(columns={'id': 'aishe_record_id', 'count_by_category_id': 'enrollment_count'})
                
                # Clean numeric columns to avoid OverflowError from infs
                int_cols = ['aishe_record_id', 'course_mode_id', 'level_id', 'programme_id', 
                            'discipline', 'course_type_id', 'enrollment_count', 'broad_discipline_group_id']
                
                for col in int_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        # Int64 is Pandas nullable integer, translates perfectly to SQLAlchemy Integer + nulls
                        df[col] = df[col].astype('Int64')

                df['survey_year'] = survey_year
                db_cols = int_cols + ['survey_year']
                df_out = df[[c for c in db_cols if c in df.columns]]
                
                # Multi-insert chunking bypasses ORM completely
                df_out.to_sql('aishe_enrollment', con=engine, if_exists='append', index=False, method='multi', chunksize=20000)
                
                count = len(df_out)
                total += count
                print(f"  ✓ {count} rows from {f}")

    print(f"  → Total enrollment rows: {total}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. NEWS CORPUS (BULK FAST)
# ──────────────────────────────────────────────────────────────────────────────

def load_news_corpus(session):
    print("\n[8/8] Loading news_corpus (BULK)...")
    
    # 1. India News Headlines (3.8M rows)
    india_path = os.path.join(DATA_DIR, "kaggle", "news", "india_news_headlines", "india-news-headlines.csv")
    total_india = 0
    print("  Loading india-news-headlines.csv into memory and deduplicating...")
    df = pd.read_csv(india_path, encoding="utf-8", low_memory=False)
    df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d', errors='coerce').dt.date
    df = df.dropna(subset=['publish_date', 'headline_text'])
    
    df = df.rename(columns={'headline_category': 'category', 'headline_text': 'headline'})
    df['category'] = df['category'].str.slice(0, 100)
    df['headline'] = df['headline'].str.slice(0, 500)
    df['source_name'] = 'india_headlines'
    
    df_out = df[['publish_date', 'source_name', 'category', 'headline']]
    df_out = df_out.drop_duplicates(subset=['publish_date', 'source_name', 'headline'])
    
    print(f"  Inserting {len(df_out):,} deduplicated rows...")
    df_out.to_sql('news_corpus', con=engine, if_exists='append', index=False, method='multi', chunksize=15000)
    print(f"  ✓ {len(df_out):,} rows from india-news-headlines.csv")

    # 2. Economic Times
    et_base = os.path.join(DATA_DIR, "kaggle", "news", "et_headlines_2022_2025")
    total_et = 0
    for year in [2022, 2023, 2024, 2025]:
        et_path = os.path.join(et_base, f"economic_times_headlines_{year}.csv")
        if not os.path.exists(et_path): continue
        df = pd.read_csv(et_path, encoding="utf-8")
        
        df['publish_date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', errors='coerce')
        # fallback parsing
        if df['publish_date'].isna().any():
            df['publish_date'] = df['publish_date'].fillna(pd.to_datetime(df['Date'], dayfirst=True, errors='coerce'))
            
        df['publish_date'] = df['publish_date'].dt.date
        df = df.dropna(subset=['publish_date', 'Headline'])
        
        df = df.rename(columns={'Headline': 'headline', 'Headline link': 'url'})
        df['headline'] = df['headline'].str.slice(0, 500)
        df['url'] = df['url'].str.slice(0, 500)
        df['source_name'] = 'economic_times'
        
        df_out = df[['publish_date', 'source_name', 'headline', 'url']]
        df_out = df_out.drop_duplicates(subset=['publish_date', 'source_name', 'headline'])
        df_out.to_sql('news_corpus', con=engine, if_exists='append', index=False, method='multi', chunksize=15000)
        
        total_et += len(df_out)
        print(f"  ✓ {len(df_out):,} rows from ET {year}")

    # 3. Financial News
    fn_path = os.path.join(DATA_DIR, "kaggle", "news", "financial_news_2003_2020", "IndianFinancialNews.csv")
    df = pd.read_csv(fn_path, encoding="utf-8")
    
    # Strip day of week manually or use regex to parse "Month DD, YYYY"
    df['clean_date'] = df['Date'].astype(str).str.replace(r', \w+$', '', regex=True)
    df['publish_date'] = pd.to_datetime(df['clean_date'], format='%B %d, %Y', errors='coerce').dt.date
    df = df.dropna(subset=['publish_date', 'Title'])
    
    df = df.rename(columns={'Title': 'headline', 'Description': 'description'})
    df['headline'] = df['headline'].str.slice(0, 500)
    df['source_name'] = 'financial_news'
    
    df_out = df[['publish_date', 'source_name', 'headline', 'description']]
    df_out = df_out.drop_duplicates(subset=['publish_date', 'source_name', 'headline'])
    df_out.to_sql('news_corpus', con=engine, if_exists='append', index=False, method='multi', chunksize=15000)
    print(f"  ✓ {len(df_out):,} rows from IndianFinancialNews.csv")

    # 4 & 5. GDELT / NewsAPI catalogs (small, keep as objects)
    gdelt_path = os.path.join(DATA_DIR, "metadata", "gdelt_catalog.json")
    gdelt_rows = []
    if os.path.exists(gdelt_path):
        with open(gdelt_path) as f:
            for _, qdata in json.load(f).get("queries", {}).items():
                for art in qdata.get("articles", []):
                    dt = pd.to_datetime(str(art.get("seendate", ""))[:8], format='%Y%m%d', errors='coerce')
                    if pd.notna(dt) and art.get("title"):
                        gdelt_rows.append({
                            'publish_date': dt.date(), 'source_name': 'gdelt',
                            'headline': art.get("title"), 'url': art.get("url")
                        })
    if gdelt_rows:
        pd.DataFrame(gdelt_rows).to_sql('news_corpus', con=engine, if_exists='append', index=False)
        print(f"  ✓ {len(gdelt_rows)} rows from GDELT catalog")

    newsapi_path = os.path.join(DATA_DIR, "metadata", "news_api_catalog.json")
    news_rows = []
    if os.path.exists(newsapi_path):
        with open(newsapi_path) as f:
            for _, qdata in json.load(f).get("queries", {}).items():
                for art in qdata.get("articles", []):
                    dt = pd.to_datetime(str(art.get("published_at", ""))[:10], format='%Y-%m-%d', errors='coerce')
                    if pd.notna(dt) and art.get("title"):
                        news_rows.append({
                            'publish_date': dt.date(), 'source_name': 'newsapi',
                            'headline': art.get("title"), 'description': art.get("description"), 'url': art.get("url")
                        })
    if news_rows:
        pd.DataFrame(news_rows).to_sql('news_corpus', con=engine, if_exists='append', index=False)
        print(f"  ✓ {len(news_rows)} rows from NewsAPI catalog")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  NEP Database — FAST Bulk Data Loading Pipeline")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Re-create tables entirely to avoid UNIQUE constraints triggering from incomplete previous runs
    print("  Ensuring clean slate (dropping and rebuilding tables)...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        try:
            load_state_reference(session)
            inst_map = load_institutions(session)
            session.commit() # Commit dimension tables
            
            # The remaining tables are loaded via direct engine connections (to_sql)
            load_nirf_rankings(session, inst_map)
            load_placements(session, inst_map)
            load_macro_controls(session)
            load_aishe_infrastructure(session)
            load_aishe_enrollment(session)
            load_news_corpus(session)

            print("\n" + "=" * 70)
            print("  ✅ ALL BULK DATA LOADED SUCCESSFULLY")
            print("=" * 70)

        except Exception as e:
            session.rollback()
            print(f"\n  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
