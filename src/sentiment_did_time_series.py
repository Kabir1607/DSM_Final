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

# Define Keyword Dictionaries for each Metric to isolate specific sentiments
METRIC_KEYWORDS = {
    'go_score': ['placement', 'job', 'jobs', 'salary', 'hiring', 'recruitment', 'unemployment', 'graduate', 'career'],
    'tlr_score': ['infrastructure', 'digital', 'classroom', 'funding', 'teacher', 'faculty', 'facility', 'budget', 'internet'],
    'oi_score': ['women', 'reservation', 'sc/st', 'minority', 'rural', 'disabled', 'scholarship', 'equity', 'inclusion'],
    'rpc_score': ['research', 'patent', 'publication', 'phd', 'innovation', 'grant', 'journal', 'science', 'development']
}

METRIC_TITLES = {
    'go_score': 'Graduation Outcomes (GO)',
    'tlr_score': 'Teaching, Learning & Resources (TLR)',
    'oi_score': 'Outreach & Inclusivity (OI)',
    'rpc_score': 'Research & Professional Practice (RPC)'
}

def get_state(text_val):
    """Helper to classify states based on text"""
    val = str(text_val).lower()
    if 'karnataka' in val or 'bangalore' in val or 'bengaluru' in val: 
        return 'Karnataka'
    if 'tamil nadu' in val or 'chennai' in val: 
        return 'Tamil Nadu'
    return 'Other'

def analyze_sentiment_correlations():
    print("==================================================")
    print("  Time-Series Analysis: Domain-Specific Sentiment")
    print("==================================================")
    
    # 1. Fetch Scores Data
    print("Fetching Institutional Scores from DB...")
    query_scores = text("""
        SELECT n.year, n.go_score, n.tlr_score, n.oi_score, n.rpc_score, i.state
        FROM nirf_rankings n
        JOIN institutions i ON n.institution_id = i.institution_id
        WHERE i.state ILIKE '%karnataka%' OR i.state ILIKE '%tamil nadu%'
    """)
    df_scores = pd.read_sql(query_scores, engine)
    df_scores['state'] = df_scores['state'].apply(get_state)
    df_scores = df_scores[df_scores['state'].isin(['Karnataka', 'Tamil Nadu'])]
    
    # 2. Fetch News Data
    print("Fetching News Sentiment Data from DB...")
    query_news = text("""
        SELECT publish_date, roberta_sentiment_score, headline
        FROM news_corpus
        WHERE roberta_sentiment_score IS NOT NULL
    """)
    df_news = pd.read_sql(query_news, engine)
    
    if df_news.empty or df_scores.empty:
        print("❌ Missing required data in database.")
        return
        
    df_news['publish_date'] = pd.to_datetime(df_news['publish_date'])
    df_news['year'] = df_news['publish_date'].dt.year
    df_news['state'] = df_news['headline'].apply(get_state)
    df_news = df_news[df_news['state'].isin(['Karnataka', 'Tamil Nadu'])]
    df_news['headline_lower'] = df_news['headline'].str.lower()
    
    # 3. Analyze Each Metric
    for metric_col, keywords in METRIC_KEYWORDS.items():
        if metric_col not in df_scores.columns:
            continue
            
        title = METRIC_TITLES[metric_col]
        print(f"\nProcessing: {title}")
        
        # Filter news for specific keywords related to this metric
        pattern = '|'.join([rf"\b{kw}\b" for kw in keywords])
        df_metric_news = df_news[df_news['headline_lower'].str.contains(pattern, na=False, regex=True)]
        
        if df_metric_news.empty:
            print(f"  -> No specific news found for {metric_col} keywords.")
            continue
            
        # Aggregate Yearly Sentiment
        yearly_sentiment = df_metric_news.groupby(['year', 'state'])['roberta_sentiment_score'].mean().reset_index()
        yearly_sentiment.rename(columns={'roberta_sentiment_score': 'avg_sentiment'}, inplace=True)
        
        # Aggregate Yearly Scores
        yearly_scores = df_scores.groupby(['year', 'state'])[metric_col].mean().reset_index()
        
        # Merge Data on Year and State
        merged_df = pd.merge(yearly_scores, yearly_sentiment, on=['year', 'state'], how='inner')
        
        if merged_df.empty or len(merged_df) < 3:
            print(f"  -> Not enough overlapping time-series data to correlate {metric_col}.")
            continue
            
        # Calculate Pearson Correlation (r)
        correlation = merged_df[metric_col].corr(merged_df['avg_sentiment'])
        print(f"  -> Correlation (r) for {metric_col}: {correlation:.3f}")
        
        # ---------------------------------------------------------
        # Plotting Dual-Axis Time Series
        # ---------------------------------------------------------
        # We will plot just Karnataka to keep the trendlines clean, as it is the Treatment state
        plot_data = merged_df[merged_df['state'] == 'Karnataka'].sort_values('year')
        
        if plot_data.empty:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        # Axis 1: The Actual NIRF Score (Left Y-Axis)
        color1 = '#2E86C1'
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Average {title}', color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(plot_data['year'], plot_data[metric_col], color=color1, marker='o', linewidth=3, label='Actual Score')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(plot_data['year'].unique())
        
        # Axis 2: The Specific Sentiment (Right Y-Axis)
        ax2 = ax1.twinx()  
        color2 = '#E74C3C'
        ax2.set_ylabel('Specific Sentiment Score', color=color2, fontsize=12, fontweight='bold')  
        line2 = ax2.plot(plot_data['year'], plot_data['avg_sentiment'], color=color2, marker='s', linestyle='--', linewidth=2.5, label='News Sentiment')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add 0-line for Sentiment context (Neutrality)
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.6)
        
        # Title and Legends
        plt.title(f"{title} vs. Domain-Specific Sentiment (Karnataka)\nPearson Correlation: r = {correlation:.3f}", fontsize=14, fontweight='bold', pad=15)
        
        # Combine legends from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True)
        
        fig.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"ts_sentiment_corr_{metric_col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  ✅ Saved Visualization: {save_path}")

if __name__ == "__main__":
    analyze_sentiment_correlations()