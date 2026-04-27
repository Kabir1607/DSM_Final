import os
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

# Force output specifically to the 'src' directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__)) if '__file__' in globals() else "src"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Database Connection
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)

def run_did_analysis_go_score():
    print("==================================================")
    print("  Difference-in-Differences (DiD) Analysis: GO Score")
    print("==================================================")
    
    print("Fetching Graduation Outcomes (GO Score) Data from DB...")
    
    # Query tailored explicitly for go_score and state filtering
    query = text("""
        SELECT n.year, n.go_score, i.state
        FROM nirf_rankings n
        JOIN institutions i ON n.institution_id = i.institution_id
        WHERE n.go_score IS NOT NULL 
          AND (i.state ILIKE '%karnataka%' OR i.state ILIKE '%tamil nadu%')
    """)
    df_inst = pd.read_sql(query, engine)
    
    if df_inst.empty:
        print("❌ No GO Score data found in the database for the specified states.")
        return
        
    # Normalize state names for plotting
    df_inst['state'] = df_inst['state'].apply(lambda x: 'Karnataka' if 'karnataka' in str(x).lower() else 'Tamil Nadu')
    
    # Exclude 2020 and 2021 (COVID-19 Gap)
    df_inst = df_inst[~df_inst['year'].isin([2020, 2021])]
    
    # Define DiD Policy Periods
    df_inst['period'] = np.where(df_inst['year'] < 2020, 'Before NEP (Pre-2020)', 'After NEP (Post-2021)')
    
    col = 'go_score'
    title = 'Graduation Outcomes (GO Score)'
    
    # Aggregate yearly means for the Time-Series plot
    yearly_avg = df_inst.groupby(['year', 'state'])[col].mean().reset_index()
    
    # ---------------------------------------------------------
    # Plot 1: TN vs KA Time-Series
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=yearly_avg, x='year', y=col, hue='state', marker='o', 
                 palette={'Karnataka': '#E74C3C', 'Tamil Nadu': '#3498DB'}, linewidth=2.5)
    
    # Shaded region for COVID exclusion & NEP Implementation line
    plt.axvspan(2019.5, 2021.5, color='gray', alpha=0.2, label='COVID-19 Gap (Excluded)')
    plt.axvline(2019.5, color='black', linestyle='--', linewidth=2, label='NEP Implementation')
    
    plt.title(f"DiD Over Time: {title}\nKarnataka (Treatment) vs Tamil Nadu (Control)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Average GO Score", fontsize=12, fontweight='bold')
    plt.legend(title='State', fontsize=10)
    plt.tight_layout()
    
    ts_path = os.path.join(OUTPUT_DIR, f"did_timeseries_{col}.png")
    plt.savefig(ts_path, dpi=300)
    plt.close()
    print(f"✅ Saved Time-Series Plot: {ts_path}")
    
    # ---------------------------------------------------------
    # Calculate DiD Estimator
    # ---------------------------------------------------------
    means = df_inst.groupby(['state', 'period'])[col].mean().unstack()
    if 'Before NEP (Pre-2020)' in means.columns and 'After NEP (Post-2021)' in means.columns:
        ka_diff = means.loc['Karnataka', 'After NEP (Post-2021)'] - means.loc['Karnataka', 'Before NEP (Pre-2020)']
        tn_diff = means.loc['Tamil Nadu', 'After NEP (Post-2021)'] - means.loc['Tamil Nadu', 'Before NEP (Pre-2020)']
        did_estimator = ka_diff - tn_diff
        
        print("\n--- Empirical Results ---")
        print(f"  Karnataka Change (Treatment): {ka_diff:.2f}")
        print(f"  Tamil Nadu Change (Control):  {tn_diff:.2f}")
        print(f"  DiD Estimator:                {did_estimator:.2f}")
        print("-------------------------\n")
        
        # ---------------------------------------------------------
        # Plot 2: Before vs After Bar Chart (Karnataka Only)
        # ---------------------------------------------------------
        ka_data = df_inst[df_inst['state'] == 'Karnataka']
        plt.figure(figsize=(8, 6))
        sns.barplot(data=ka_data, x='period', y=col, errorbar=None, palette=['#D35400', '#27AE60'], order=['Before NEP (Pre-2020)', 'After NEP (Post-2021)'])
        plt.title(f"Karnataka Before vs After NEP: {title}", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Average GO Score", fontsize=12, fontweight='bold')
        plt.xlabel("Policy Period", fontsize=12, fontweight='bold')
        
        # Annotate exact values onto the bars
        for p in plt.gca().patches:
            val = p.get_height()
            if not pd.isna(val):
                plt.gca().annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val), 
                                   ha='center', va='center', xytext=(0, -15), 
                                   textcoords='offset points', fontsize=12, fontweight='bold', color='white')
        
        plt.tight_layout()
        ka_bar_path = os.path.join(OUTPUT_DIR, f"ka_before_after_{col}.png")
        plt.savefig(ka_bar_path, dpi=300)
        plt.close()
        print(f"✅ Saved Karnataka Policy Chart: {ka_bar_path}")

        # ---------------------------------------------------------
        # Plot 3: Difference-in-Differences Bar Chart (TN vs KA)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_inst, x='state', y=col, hue='period', errorbar=None, palette=['#BDC3C7', '#2E86C1'], order=['Tamil Nadu', 'Karnataka'])
        plt.title(f"Difference-in-Differences: {title}", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel(f"Average {title}", fontsize=12, fontweight='bold')
        plt.xlabel("State Cohort", fontsize=12, fontweight='bold')
        plt.legend(title='Period', fontsize=10)
        
        plt.tight_layout()
        did_bar_path = os.path.join(OUTPUT_DIR, f"did_bar_{col}.png")
        plt.savefig(did_bar_path, dpi=300)
        plt.close()
        print(f"✅ Saved DiD State Comparison Chart: {did_bar_path}")

if __name__ == "__main__":
    run_did_analysis_go_score()