"""
basic_EDA.py
============
Exploratory Data Analysis script to summarize metadata, ranges, 
missing values (NaNs), and basic correlations for the PostgreSQL database.
Memory-optimized to only analyze data relevant to the DiD analysis.
"""

import os
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

# Setup paths for saving EDA reports and plots
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EDA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "EDA_Reports")
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)

# Only pull the exact data needed for the DiD analysis to save RAM
DID_QUERIES = {
    'institutions_target': text("SELECT * FROM institutions WHERE state ILIKE '%karnataka%' OR state ILIKE '%tamil nadu%'"),
    'nirf_rankings_target': text("""
        SELECT n.* 
        FROM nirf_rankings n
        JOIN institutions i ON n.institution_id = i.institution_id
        WHERE i.state ILIKE '%karnataka%' OR i.state ILIKE '%tamil nadu%'
    """),
    'placements_target': text("""
        SELECT p.* 
        FROM placements p
        JOIN institutions i ON p.institution_id = i.institution_id
        WHERE i.state ILIKE '%karnataka%' OR i.state ILIKE '%tamil nadu%'
    """),
    'macro_controls': text("SELECT * FROM macro_controls"),
    'news_sentiment': text("SELECT * FROM news_corpus WHERE roberta_sentiment_score IS NOT NULL")
}

def extract_metadata():
    """
    Extracts metadata, missing values, and ranges only for the target datasets.
    """
    print("\n--- Extracting Targeted Database Metadata ---")
    metadata_summary = {}
    
    for name, query in DID_QUERIES.items():
        print(f"Analyzing dataset: {name}...")
        df = pd.read_sql(query, engine)
        
        # 1. Column Types & Missing Values
        info_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Values (NaN)': df.isnull().sum(),
            '% Missing': (df.isnull().sum() / len(df) * 100).round(2) if len(df) > 0 else 0
        })
        
        # 2. Ranges (Min/Max) for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and not df.empty:
            desc = df[numeric_cols].describe().T
            info_df['Min'] = desc['min']
            info_df['Max'] = desc['max']
            info_df['Mean'] = desc['mean'].round(2)
        else:
            info_df['Min'] = np.nan
            info_df['Max'] = np.nan
            info_df['Mean'] = np.nan
            
        csv_path = os.path.join(EDA_OUTPUT_DIR, f"{name}_metadata.csv")
        info_df.to_csv(csv_path)
        print(f"  -> Saved {csv_path} (Rows: {len(df)})")
        
        # Keep DataFrame in memory for the correlations function
        metadata_summary[name] = df
        
    return metadata_summary

def generate_correlations_and_plots(metadata_summary):
    """
    Generates correlation matrices and basic visualizations for the target subsets.
    """
    print("\n--- Generating Correlations & Visualizations ---")
    
    # 1. NIRF Rankings Analysis
    df_nirf = metadata_summary.get('nirf_rankings_target', pd.DataFrame())
    if not df_nirf.empty:
        cols_to_corr = [c for c in df_nirf.columns if c not in ['ranking_id', 'institution_id', 'year']]
        numeric_nirf = df_nirf[cols_to_corr].select_dtypes(include=[np.number]).dropna()
        if len(numeric_nirf) > 0:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_nirf.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Matrix: Target NIRF Metrics (KA & TN)", pad=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, "nirf_target_correlation.png"), dpi=300)
            plt.close()
            print("  -> Saved nirf_target_correlation.png")
            
            # Distribution of GO Scores
            plt.figure(figsize=(9, 5))
            sns.histplot(df_nirf['go_score'].dropna(), kde=True, bins=30, color='indigo')
            plt.title("Distribution of target Graduation Outcomes (GO Score)", fontweight='bold')
            plt.xlabel("GO Score")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, "nirf_target_go_score_dist.png"), dpi=300)
            plt.close()
            print("  -> Saved nirf_target_go_score_dist.png")

    # 2. Placements Analysis
    df_placements = metadata_summary.get('placements_target', pd.DataFrame())
    if not df_placements.empty:
        cols_to_corr = [c for c in df_placements.columns if c not in ['placement_id', 'institution_id']]
        numeric_place = df_placements[cols_to_corr].select_dtypes(include=[np.number]).dropna()
        if len(numeric_place) > 0:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_place.corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Matrix: Target Placements (KA & TN)", pad=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, "placements_target_correlation.png"), dpi=300)
            plt.close()
            print("  -> Saved placements_target_correlation.png")

def main():
    print("==================================================")
    print("   Targeted Exploratory Data Analysis (DiD)")
    print("==================================================")
    
    metadata = extract_metadata()
    generate_correlations_and_plots(metadata)
    
    print("\nEDA Completed successfully without exploding RAM! Check the 'data/processed/EDA_Reports' folder.")

if __name__ == "__main__":
    main()
