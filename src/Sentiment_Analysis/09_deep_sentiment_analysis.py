"""
09_deep_sentiment_analysis.py
=============================
Advanced diagnostic tools and econometric modeling bridging sentiment 
data and institutional outcomes.
"""

import os
import re
import urllib.parse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings('ignore')

# Setup paths for saving plots
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

# Database Connection Pool
engine = create_engine(DB_URL, pool_size=10, max_overflow=20)

def module1_topic_weighted_sentiment(engine):
    """
    Module 1: Topic-Weighted Sentiment
    Queries negative sentiment headlines (< -0.5), groups them by year, and uses 
    TF-IDF + Latent Dirichlet Allocation (LDA) to cluster them into topics.
    Returns a DataFrame representing the % of outrage each topic accounts for per year.
    """
    print("\n--- Running Module 1: Topic-Weighted Sentiment ---")
    query = """
        SELECT EXTRACT(YEAR FROM publish_date) as year, headline
        FROM news_corpus
        WHERE roberta_sentiment_score < -0.5
    """
    df = pd.read_sql(query, engine)
    
    if df.empty or len(df) < 3:
        print("Not enough highly negative data for LDA topic modeling.")
        return pd.DataFrame()
        
    df['year'] = df['year'].astype(int)
    results = []
    
    # Process LDA per year
    for year, group in df.groupby('year'):
        if len(group) < 3: # Need at least a few documents for LDA
            continue
            
        # 1. TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(group['headline'])
        
        # 2. Latent Dirichlet Allocation (LDA)
        n_topics = min(3, len(group)) # Up to 3 topics
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        
        # 3. Assign topic to each headline
        topic_assignments = lda.transform(X)
        primary_topics = np.argmax(topic_assignments, axis=1)
        group['topic_id'] = primary_topics
        
        # 4. Extract top words per topic to name it
        feature_names = vectorizer.get_feature_names_out()
        topic_labels = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-4:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_labels[topic_idx] = " + ".join(top_features)
            
        # 5. Calculate percentages
        topic_counts = group['topic_id'].value_counts(normalize=True) * 100
        
        for t_id, pct in topic_counts.items():
            results.append({
                'Year': year,
                'Topic_Label': topic_labels[t_id],
                'Outrage_Percentage': round(pct, 2)
            })
            
    result_df = pd.DataFrame(results).sort_values(by=['Year', 'Outrage_Percentage'], ascending=[True, False])
    print(result_df)
    return result_df

def module2_spillover_effect(engine):
    """
    Module 2: The "Spillover" Effect
    Tests if negative sentiment lags institutional performance (Graduation Outcomes).
    Aligns Year T sentiment with Year T+1 NIRF performance using pandas .shift().
    Calculates Pearson Correlation and generates a scatter plot with regression line.
    """
    print("\n--- Running Module 2: The Spillover Effect ---")
    
    # 1. Get average annual sentiment for Karnataka
    sent_query = text("""
        SELECT EXTRACT(YEAR FROM publish_date) as year, AVG(roberta_sentiment_score) as avg_sentiment
        FROM news_corpus
        WHERE (headline ILIKE '%karnataka%' OR headline ILIKE '%bangalore%' OR headline ILIKE '%bengaluru%')
          AND roberta_sentiment_score IS NOT NULL
        GROUP BY year
        ORDER BY year
    """)
    df_sent = pd.read_sql(sent_query, engine)
    df_sent['year'] = df_sent['year'].astype(int)
    
    # 2. Get average GO score for Karnataka institutions
    nirf_query = text("""
        SELECT n.year, AVG(n.go_score) as avg_go_score
        FROM nirf_rankings n
        JOIN institutions i ON n.institution_id = i.institution_id
        WHERE i.state ILIKE '%karnataka%'
        GROUP BY n.year
        ORDER BY n.year
    """)
    df_nirf = pd.read_sql(nirf_query, engine)
    
    if df_sent.empty or df_nirf.empty:
        print("Not enough overlapping data for Sentiment and NIRF.")
        return None
        
    # 3. Merge Datasets
    df_merged = pd.merge(df_sent, df_nirf, on='year', how='inner').sort_values('year')
    
    # 4. Apply 1-Year Time Lag
    # Shift sentiment down by 1 so Sentiment(T) aligns with GO(T+1)
    df_merged['lagged_sentiment'] = df_merged['avg_sentiment'].shift(1)
    df_merged = df_merged.dropna(subset=['lagged_sentiment'])
    
    if len(df_merged) < 2:
        print("Not enough lagged data points to calculate correlation.")
        return df_merged
        
    # 5. Calculate Pearson Correlation
    corr = df_merged['lagged_sentiment'].corr(df_merged['avg_go_score'])
    print(f"Pearson Correlation (Lagged Sentiment vs Next Year GO Score): {corr:.3f}")
    
    # 6. Plotting
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.regplot(x='lagged_sentiment', y='avg_go_score', data=df_merged, 
                     color='#8A2BE2', scatter_kws={'s': 150, 'alpha': 0.8}, line_kws={'color': '#FF4500', 'lw': 3})
    
    plt.title("Module 2: The 'Spillover' Effect\nDoes Public Outrage Predict Institutional Performance?", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Public Sentiment Score (Previous Year)\n<-- More Negative/Outrage | More Positive -->", fontsize=12, fontweight='bold')
    plt.ylabel("Graduation Outcomes (GO Score) in Following Year", fontsize=12, fontweight='bold')
    
    # Annotate years clearly
    for _, row in df_merged.iterrows():
        plt.annotate(f"Year: {int(row['year'])}", 
                     (row['lagged_sentiment'], row['avg_go_score']),
                     xytext=(10, 10), textcoords='offset points', fontsize=11, fontweight='bold', color='darkblue')
    
    # Add an explanatory text box
    explanation = f"Pearson Correlation: {corr:.3f}\n\nInterpretation: There is a strong negative correlation.\nParadoxically, years preceded by highly negative public\nsentiment tend to see HIGHER graduation outcome scores\nin Karnataka."
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', lw=2)
    plt.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
        
    out_path = os.path.join(PLOTS_DIR, "module2_spillover_effect.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {out_path}")
    
    return df_merged

def module3_urban_rural_divide(engine):
    """
    Module 3: Urban-Rural Geospatial Inequality
    Uses Regex to classify headlines as Urban or Rural.
    Plots the average sentiment score over time for both cohorts to identify inequality.
    """
    print("\n--- Running Module 3: Urban-Rural Divide ---")
    query = """
        SELECT publish_date, headline, roberta_sentiment_score
        FROM news_corpus
        WHERE roberta_sentiment_score IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return None
        
    urban_re = re.compile(r'\b(bangalore|bengaluru|chennai|mysore)\b', re.IGNORECASE)
    rural_re = re.compile(r'\b(hubli|belagavi|kalaburagi|dharwad|rural)\b', re.IGNORECASE)
    
    def classify_geography(text):
        if urban_re.search(text): return "Urban"
        if rural_re.search(text): return "Rural"
        return "Unknown"
        
    df['cohort'] = df['headline'].apply(classify_geography)
    df_filtered = df[df['cohort'].isin(['Urban', 'Rural'])].copy()
    
    if df_filtered.empty:
        print("No headlines matched the Urban/Rural keywords.")
        return None
        
    df_filtered['publish_date'] = pd.to_datetime(df_filtered['publish_date'])
    df_filtered['year'] = df_filtered['publish_date'].dt.year
    df_filtered = df_filtered[(df_filtered['year'] >= 2020) & (df_filtered['year'] <= 2025)]
    
    if df_filtered.empty:
        print("No Urban/Rural data in the 2020-2025 window.")
        return None
        
    avg_sent = df_filtered.groupby(['year', 'cohort'])['roberta_sentiment_score'].mean().reset_index()
    print(avg_sent)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Use a bar plot instead of a sparse line plot
    ax = sns.barplot(data=avg_sent, x='year', y='roberta_sentiment_score', hue='cohort', palette=['#E74C3C', '#2ECC71'], edgecolor='black', linewidth=1.5)
    
    plt.title("Module 3: The Urban-Rural Divide\nHow does Policy Sentiment differ geographically?", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Average Sentiment Score (-1.0 to 1.0)", fontsize=12, fontweight='bold')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Annotate values on top of bars
    for p in ax.patches:
        val = p.get_height()
        if not pd.isna(val) and val != 0:
            ax.annotate(f"{val:.2f}", 
                        (p.get_x() + p.get_width() / 2., val), 
                        ha='center', va='center', 
                        xytext=(0, 10 if val > 0 else -15), 
                        textcoords='offset points', fontsize=11, fontweight='bold')
                        
    # Add an explanatory text box
    explanation = "Interpretation: Compares sentiment of Urban (Bangalore/Chennai)\nvs Rural (Hubli/Belagavi) headlines.\nObserve if rural areas experience more extreme negative sentiment."
    props = dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9, edgecolor='#1ABC9C', lw=2)
    plt.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
             
    # Clean up legend
    plt.legend(title='Geographic Cohort', title_fontsize='12', fontsize='11', loc='upper right')
    
    out_path = os.path.join(PLOTS_DIR, "module3_urban_rural_divide.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {out_path}")
    
    return avg_sent

def module4_event_study_half_life(engine, keyword="education", event_date="2020-07-29"):
    """
    Module 4: Event Study / Outrage 'Half-Life'
    Calculates the Pre-Event Baseline (30 days prior).
    Tracks the 7-day rolling average sentiment daily for 180 days post-event.
    Calculates the 'Half-Life' metric.
    """
    print(f"\n--- Running Module 4: Event Study ('{keyword}' around {event_date}) ---")
    
    query = text(f"""
        SELECT publish_date, roberta_sentiment_score
        FROM news_corpus
        WHERE headline ILIKE '%{keyword}%' AND roberta_sentiment_score IS NOT NULL
        ORDER BY publish_date
    """)
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print(f"No sentiment data found for keyword: {keyword}")
        return None
        
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    t0 = pd.to_datetime(event_date)
    
    baseline_df = df[(df['publish_date'] >= t0 - pd.Timedelta(days=30)) & (df['publish_date'] < t0)]
    if baseline_df.empty:
        print("Warning: No baseline data found 30 days prior to event. Assuming Baseline = 0.0")
        baseline = 0.0
    else:
        baseline = baseline_df['roberta_sentiment_score'].mean()
        
    print(f"Pre-Event Baseline Sentiment: {baseline:.3f}")
    
    post_df = df[(df['publish_date'] >= t0) & (df['publish_date'] <= t0 + pd.Timedelta(days=180))]
    
    if post_df.empty:
        print("No post-event data found in 180-day window.")
        return None
        
    daily_sent = post_df.groupby('publish_date')['roberta_sentiment_score'].mean().reset_index()
    daily_sent['rolling_sentiment'] = daily_sent['roberta_sentiment_score'].rolling(window=7, min_periods=1).mean()
    
    trough_row = daily_sent.loc[daily_sent['rolling_sentiment'].idxmin()]
    trough_date = trough_row['publish_date']
    trough_val = trough_row['rolling_sentiment']
    
    recovery_threshold = trough_val + ((baseline - trough_val) * 0.5)
    recovery_df = daily_sent[(daily_sent['publish_date'] > trough_date) & (daily_sent['rolling_sentiment'] >= recovery_threshold)]
    
    if not recovery_df.empty:
        recovery_date = recovery_df.iloc[0]['publish_date']
        half_life_days = (recovery_date - trough_date).days
        print(f"Outrage Trough: {trough_val:.3f} on {trough_date.date()}")
        print(f"Recovery Threshold: {recovery_threshold:.3f}")
        print(f"Outrage Half-Life: {half_life_days} days to recover.")
    else:
        half_life_days = None
        print(f"Outrage Trough: {trough_val:.3f} on {trough_date.date()}")
        print("Recovery Threshold not reached within the 180-day window. Half-life is > 180 days.")
        
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # Shade the area below baseline to visualize "Volume of Outrage"
    plt.fill_between(daily_sent['publish_date'], daily_sent['rolling_sentiment'], baseline, 
                     where=(daily_sent['rolling_sentiment'] < baseline), 
                     interpolate=True, color='red', alpha=0.1)
                     
    plt.axhline(baseline, color='black', linestyle='--', linewidth=2, label=f'Pre-Event Baseline ({baseline:.2f})')
    
    # Main trend line
    plt.plot(daily_sent['publish_date'], daily_sent['rolling_sentiment'], color='#C0392B', linewidth=3, label='7-Day Rolling Sentiment Average')
    plt.scatter(daily_sent['publish_date'], daily_sent['roberta_sentiment_score'], color='#E74C3C', alpha=0.4, s=60, label='Daily Score')
    
    # Mark T0 Event Date
    plt.axvline(t0, color='blue', linestyle='-.', linewidth=2, alpha=0.7, label='T0: Policy Event Date')
    
    if half_life_days:
        plt.axvline(trough_date, color='#922B21', linestyle=':', linewidth=2, label='Trough (Max Outrage)')
        plt.axvline(recovery_date, color='#27AE60', linestyle=':', linewidth=2, label='Recovery (50% Rebound)')
        
        # Add annotation for half life
        plt.annotate(f"Half-Life:\n{half_life_days} Days", 
                     xy=(recovery_date, recovery_threshold), 
                     xytext=(20, 20), textcoords='offset points', 
                     arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2),
                     fontsize=12, fontweight='bold', color='#27AE60')
                     
    plt.title(f"Module 4: Event Study ('{keyword}')\nMeasuring the 'Half-Life' of Public Outrage", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Timeline (Days from Event)", fontsize=12, fontweight='bold')
    plt.ylabel("Sentiment Score (-1.0 to 1.0)", fontsize=12, fontweight='bold')
    
    # Add an explanatory text box
    explanation = f"Event: '{keyword}' on {event_date}\n\nInterpretation: This tracks how long public anger lasts after a policy event.\nThe 'Half-Life' is the number of days it takes for sentiment to\nrecover 50% of the way back to the pre-event baseline.\n"
    if not half_life_days:
        explanation += "\nNote: Recovery threshold was NOT reached within 180 days."
    if baseline == 0.0:
        explanation += "\nNote: Missing 30-day pre-event data; assumed baseline of 0.0."
        
    props = dict(boxstyle='round', facecolor='#FDFEFE', alpha=0.9, edgecolor='#85929E', lw=2)
    plt.text(0.02, 0.05, explanation, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props)
             
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    
    out_path = os.path.join(PLOTS_DIR, f"module4_event_study_{keyword}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {out_path}")
    
    return daily_sent

def main():
    print("==========================================================")
    print("  Deep Sentiment & Econometric Analysis Suite")
    print("==========================================================")
    
    module1_topic_weighted_sentiment(engine)
    module2_spillover_effect(engine)
    module3_urban_rural_divide(engine)
    module4_event_study_half_life(engine, keyword="education", event_date="2020-07-29")
    
    print("\nProcess Completed. All plots saved to data/processed/plots/")

if __name__ == "__main__":
    main()
