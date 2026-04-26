"""
011_did.py
==========
Performs Difference-in-Differences (DiD) Analysis and Interrupted Time-Series Analysis
using the metrics defined in the 'Data Gaps and Variable Mapping Revision' CSV.
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DID_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "EDA_Reports", "did_analysis")
os.makedirs(DID_OUTPUT_DIR, exist_ok=True)

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)

def run_did_analysis():
    print("==================================================")
    print("  Difference-in-Differences (DiD) Analysis")
    print("==================================================")
    
    # 1. Fetch NIRF & Placements Data
    print("Fetching Institutional Data from DB...")
    query_institutions = text("""
        SELECT n.year, n.tlr_score, n.oi_score, n.rpc_score, n.overall_score, 
               p.avg_salary_lpa, i.state
        FROM nirf_rankings n
        JOIN institutions i ON n.institution_id = i.institution_id
        LEFT JOIN placements p ON n.institution_id = p.institution_id AND n.year = p.year
        WHERE i.state ILIKE '%karnataka%' OR i.state ILIKE '%tamil nadu%'
    """)
    df_inst = pd.read_sql(query_institutions, engine)
    
    # Normalize state names
    df_inst['state'] = df_inst['state'].apply(lambda x: 'Karnataka' if 'karnataka' in str(x).lower() else 'Tamil Nadu')
    
    # Exclude 2020 and 2021 (COVID-19 Years)
    df_inst = df_inst[~df_inst['year'].isin([2020, 2021])]
    
    # Determine Before vs After (Pre-2020 = Before, Post-2021 = After)
    df_inst['period'] = np.where(df_inst['year'] < 2020, 'Before NEP (Pre-2020)', 'After NEP (Post-2021)')
    
    metrics = {
        'tlr_score': 'Infrastructure & Resources (TLR Score)',
        'oi_score': 'Inclusivity & Equity (OI Score)',
        'rpc_score': 'Research Restructuring (RPC Score)',
        'avg_salary_lpa': 'Direct Placement Data (Avg Salary LPA)'
    }
    
    for col, title in metrics.items():
        if col not in df_inst.columns or df_inst[col].isnull().all():
            print(f"Skipping {title} - No data available.")
            continue
            
        print(f"\nAnalyzing: {title}")
        
        # Aggregate yearly means
        yearly_avg = df_inst.groupby(['year', 'state'])[col].mean().reset_index()
        
        # Plot 1: TN vs KA Time-Series
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=yearly_avg, x='year', y=col, hue='state', marker='o', 
                     palette={'Karnataka': '#E74C3C', 'Tamil Nadu': '#3498DB'}, linewidth=2.5)
        
        # Shaded region for COVID exclusion
        plt.axvspan(2019.5, 2021.5, color='gray', alpha=0.2, label='COVID-19 Gap (Excluded)')
        plt.axvline(2019.5, color='black', linestyle='--', linewidth=2, label='NEP Implementation')
        
        plt.title(f"DiD Over Time: {title}\nKarnataka (Treatment) vs Tamil Nadu (Control)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Year", fontsize=12, fontweight='bold')
        plt.ylabel(title, fontsize=12, fontweight='bold')
        plt.legend(title='State', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(DID_OUTPUT_DIR, f"did_timeseries_{col}.png"), dpi=300)
        plt.close()
        
        # Calculate standard DiD estimators (difference of means)
        means = df_inst.groupby(['state', 'period'])[col].mean().unstack()
        if 'Before NEP (Pre-2020)' in means.columns and 'After NEP (Post-2021)' in means.columns:
            ka_diff = means.loc['Karnataka', 'After NEP (Post-2021)'] - means.loc['Karnataka', 'Before NEP (Pre-2020)']
            tn_diff = means.loc['Tamil Nadu', 'After NEP (Post-2021)'] - means.loc['Tamil Nadu', 'Before NEP (Pre-2020)']
            did_estimator = ka_diff - tn_diff
            
            print(f"  Karnataka Change: {ka_diff:.2f}")
            print(f"  Tamil Nadu Change: {tn_diff:.2f}")
            print(f"  DiD Estimator: {did_estimator:.2f}")
            
            # Plot 2: Before vs After Bar Chart (Karnataka vs Karnataka)
            ka_data = df_inst[df_inst['state'] == 'Karnataka']
            plt.figure(figsize=(8, 6))
            sns.barplot(data=ka_data, x='period', y=col, errorbar=None, palette=['#D35400', '#27AE60'], order=['Before NEP (Pre-2020)', 'After NEP (Post-2021)'])
            plt.title(f"Karnataka Before vs After NEP: {title}", fontsize=14, fontweight='bold', pad=15)
            plt.ylabel(title, fontsize=12, fontweight='bold')
            plt.xlabel("Policy Period", fontsize=12, fontweight='bold')
            
            # Add value labels
            for p in plt.gca().patches:
                val = p.get_height()
                plt.gca().annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val), 
                                   ha='center', va='center', xytext=(0, -15), 
                                   textcoords='offset points', fontsize=12, fontweight='bold', color='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(DID_OUTPUT_DIR, f"ka_before_after_{col}.png"), dpi=300)
            plt.close()

            # Plot 3: Before vs After DiD Bar Chart (TN vs KA)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_inst, x='state', y=col, hue='period', errorbar=None, palette=['#BDC3C7', '#2E86C1'], order=['Tamil Nadu', 'Karnataka'])
            plt.title(f"Difference-in-Differences: {title}", fontsize=14, fontweight='bold', pad=15)
            plt.ylabel(f"Average {title}", fontsize=12, fontweight='bold')
            plt.xlabel("State Cohort", fontsize=12, fontweight='bold')
            plt.legend(title='Period', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(DID_OUTPUT_DIR, f"did_bar_{col}.png"), dpi=300)
            plt.close()

    # 2. Fetch News Sentiment Data (Administrative Efficiency)
    print("\nAnalyzing: Administrative Efficiency (News Sentiment)")
    query_news = text("""
        SELECT publish_date, roberta_sentiment_score, headline
        FROM news_corpus
        WHERE roberta_sentiment_score IS NOT NULL
    """)
    df_news = pd.read_sql(query_news, engine)
    
    if not df_news.empty:
        df_news['publish_date'] = pd.to_datetime(df_news['publish_date'])
        df_news['year'] = df_news['publish_date'].dt.year
        
        # State classification via regex
        def classify_state(text):
            text = text.lower()
            if 'karnataka' in text or 'bangalore' in text or 'bengaluru' in text: return 'Karnataka'
            if 'tamil nadu' in text or 'chennai' in text: return 'Tamil Nadu'
            return 'Other'
            
        df_news['state'] = df_news['headline'].apply(classify_state)
        df_news = df_news[df_news['state'].isin(['Karnataka', 'Tamil Nadu'])]
        
        # Exclude COVID years
        df_news = df_news[~df_news['year'].isin([2020, 2021])]
        
        df_news['period'] = np.where(df_news['year'] < 2020, 'Before NEP (Pre-2020)', 'After NEP (Post-2021)')
        
        yearly_news = df_news.groupby(['year', 'state'])['roberta_sentiment_score'].mean().reset_index()
        
        if not yearly_news.empty:
            plt.figure(figsize=(12, 6))
            sns.set_theme(style="whitegrid")
            sns.lineplot(data=yearly_news, x='year', y='roberta_sentiment_score', hue='state', marker='o', 
                         palette={'Karnataka': '#E74C3C', 'Tamil Nadu': '#3498DB'}, linewidth=2.5)
            plt.axvspan(2019.5, 2021.5, color='gray', alpha=0.2, label='COVID-19 Gap (Excluded)')
            plt.axhline(0, color='black', linestyle=':', alpha=0.5)
            plt.title("Administrative Efficiency: Public Sentiment Score Over Time\nKarnataka (Treatment) vs Tamil Nadu (Control)", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Year", fontsize=12, fontweight='bold')
            plt.ylabel("Avg Sentiment Score", fontsize=12, fontweight='bold')
            plt.legend(title='State')
            plt.tight_layout()
            plt.savefig(os.path.join(DID_OUTPUT_DIR, f"did_timeseries_sentiment.png"), dpi=300)
            plt.close()

def main():
    run_did_analysis()
    print("\nVisualizations saved to: data/processed/EDA_Reports/did_analysis")

if __name__ == "__main__":
    main()
