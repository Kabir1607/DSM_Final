"""
010_data_triangulation.py
=========================
Performs Data Triangulation testing to find anomalies and inconsistencies 
between government reported indices (NIRF) and raw ground truth data (Placements/AISHE).
Generates visualizations of the overlaps.
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
TRIANGULATION_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "EDA_Reports", "data_triangulation_results")
os.makedirs(TRIANGULATION_DIR, exist_ok=True)
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)
def run_test_1_placement_quality(engine):
    """
    Test 1: Placement (Quantity vs Quality)
    Matches NIRF Graduation Outcomes (GO) vs Ground Truth Average Placement Salary (LPA).
    Hypothesis: High GO but Low LPA indicates weighting quantity over quality.
    """
    print("\n--- Running Test 1: Placement (Quantity vs. Quality) ---")
    query = text("""
        SELECT n.institution_id, n.year, n.go_score, p.avg_salary_lpa
        FROM nirf_rankings n
        JOIN placements p ON n.institution_id = p.institution_id
        WHERE p.avg_salary_lpa IS NOT NULL AND n.go_score IS NOT NULL
    """)
    df = pd.read_sql(query, engine)
    
    if len(df) < 2:
        print("Not enough data to triangulate Placements vs NIRF GO.")
        return
        
    corr = df['go_score'].corr(df['avg_salary_lpa'])
    print(f"Correlation (GO Score vs Avg Salary LPA): {corr:.3f}")
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x='go_score', y='avg_salary_lpa', data=df, color='#3498DB', s=100, alpha=0.7, edgecolor='black')
    
    # Add trend line
    sns.regplot(x='go_score', y='avg_salary_lpa', data=df, scatter=False, color='#E74C3C', line_kws={"lw": 2})
    
    plt.title(f"Triangulation Test 1: NIRF GO Score vs. Actual Average Salary\nPearson r = {corr:.2f}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("NIRF Graduation Outcomes (GO) Score", fontsize=12, fontweight='bold')
    plt.ylabel("Average Placement Salary (LPA)", fontsize=12, fontweight='bold')
    
    # Text box for insight
    explanation = "Insight: Look for dots in the bottom-right quadrant.\nThese are institutions with HIGH government scores\nbut surprisingly LOW actual salaries (quantity over quality)."
    plt.text(0.02, 0.95, explanation, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
             
    plt.tight_layout()
    plt.savefig(os.path.join(TRIANGULATION_DIR, "test1_go_vs_lpa.png"), dpi=300)
    plt.close()
def run_test_2_infrastructure_integrity(engine):
    """
    Test 2: Infrastructure (Faculty Integrity)
    Matches NIRF TLR Score (Teaching, Learning & Resources) vs Ground Truth Student-Faculty Ratio.
    Hypothesis: TLR should negatively correlate with Student-Faculty Ratio. 
    """
    print("\n--- Running Test 2: Infrastructure Integrity (TLR vs PTR) ---")
    query = text("""
        SELECT n.institution_id, n.year, n.tlr_score, p.student_faculty_ratio
        FROM nirf_rankings n
        JOIN placements p ON n.institution_id = p.institution_id
        WHERE p.student_faculty_ratio IS NOT NULL AND n.tlr_score IS NOT NULL
    """)
    df = pd.read_sql(query, engine)
    
    if len(df) < 2:
        print("Not enough data to triangulate Infrastructure vs TLR.")
        return
        
    corr = df['tlr_score'].corr(df['student_faculty_ratio'])
    print(f"Correlation (TLR Score vs Student-Faculty Ratio): {corr:.3f}")
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x='student_faculty_ratio', y='tlr_score', data=df, color='#2ECC71', s=100, alpha=0.7, edgecolor='black')
    sns.regplot(x='student_faculty_ratio', y='tlr_score', data=df, scatter=False, color='#8E44AD', line_kws={"lw": 2})
    
    plt.title(f"Triangulation Test 2: NIRF TLR Score vs. Student-Faculty Ratio\nPearson r = {corr:.2f}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Student-Faculty Ratio (PTR)", fontsize=12, fontweight='bold')
    plt.ylabel("NIRF TLR (Teaching & Resources) Score", fontsize=12, fontweight='bold')
    
    explanation = "Insight: We expect a negative trend (higher ratio = lower score).\nIf points exist in the top-right (High Ratio AND High Score),\nit suggests manipulation of faculty counts."
    plt.text(0.98, 0.95, explanation, transform=plt.gca().transAxes, fontsize=11,
             horizontalalignment='right', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.9, edgecolor='blue'))
             
    plt.tight_layout()
    plt.savefig(os.path.join(TRIANGULATION_DIR, "test2_tlr_vs_ptr.png"), dpi=300)
    plt.close()
def run_test_4_cost_vs_quality(engine):
    """
    Test 4: Cost vs Quality (Equity Gap)
    Groups institutions by High/Low Fees and compares their NIRF Overall Scores.
    """
    print("\n--- Running Test 4: Cost vs Quality (Equity Gap) ---")
    query = text("""
        SELECT n.institution_id, n.year, n.overall_score, p.fees_ug_inr
        FROM nirf_rankings n
        JOIN placements p ON n.institution_id = p.institution_id
        WHERE p.fees_ug_inr IS NOT NULL AND n.overall_score IS NOT NULL
    """)
    df = pd.read_sql(query, engine)
    
    if len(df) < 2:
        print("Not enough data to triangulate Fees vs Overall Score.")
        return
        
    # Create fee buckets (median split)
    median_fee = df['fees_ug_inr'].median()
    df['fee_bucket'] = np.where(df['fees_ug_inr'] > median_fee, 'High Fee', 'Low Fee')
    
    # Calculate medians for plot
    medians = df.groupby('fee_bucket')['overall_score'].median().reset_index()
    print("Median NIRF Scores by Fee Bucket:")
    print(medians)
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    # Boxplot for distribution
    sns.boxplot(x='fee_bucket', y='overall_score', data=df, palette=['#F39C12', '#9B59B6'], width=0.5)
    
    plt.title("Triangulation Test 4: Cost vs Quality (Equity Gap)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Institution Cost Bracket", fontsize=12, fontweight='bold')
    plt.ylabel("NIRF Overall Score", fontsize=12, fontweight='bold')
    
    explanation = f"Insight: Median split point is ₹{median_fee:,.0f}.\nIf High Fee schools systematically outperform Low Fee schools,\nit indicates that the multidisciplinary metrics favor well-funded institutions."
    plt.text(0.02, 0.05, explanation, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='#FDFEFE', alpha=0.9, edgecolor='gray'))
             
    plt.tight_layout()
    plt.savefig(os.path.join(TRIANGULATION_DIR, "test4_fees_vs_score.png"), dpi=300)
    plt.close()
def main():
    print("==========================================================")
    print("  Data Triangulation: Validation Testing Suite")
    print("==========================================================")
    
    run_test_1_placement_quality(engine)
    run_test_2_infrastructure_integrity(engine)
    run_test_4_cost_vs_quality(engine)
    
    print("\nTriangulation Visualizations Saved to:")
    print(TRIANGULATION_DIR)
if __name__ == "__main__":
    main()