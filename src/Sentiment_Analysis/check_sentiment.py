"""
check_sentiment.py
==================
Checks the Database to verify that RoBERTa sentiment scores 
were correctly inserted into the `news_corpus` table.
"""

import urllib.parse
from sqlalchemy import create_engine
import pandas as pd

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

def main():
    engine = create_engine(DB_URL)
    
    print("==================================================")
    print("   Sentiment Analysis Database Verification")
    print("==================================================")
    
    # 1. Check Total Scored Rows
    query_count = "SELECT COUNT(*) FROM news_corpus WHERE roberta_sentiment_score IS NOT NULL"
    total_scored = pd.read_sql(query_count, engine).iloc[0, 0]
    print(f"Total Scored Headlines in Database: {total_scored}")
    
    if total_scored == 0:
        print("\nERROR: No sentiment scores found! You might need to run the sentiment script first.")
        return
        
    # 2. Show 5 Most Positive Headlines
    print("\n--- Top 5 Most POSITIVE Headlines (+1.0) ---")
    query_pos = """
    SELECT publish_date, roberta_sentiment_score, headline 
    FROM news_corpus 
    WHERE roberta_sentiment_score IS NOT NULL 
    ORDER BY roberta_sentiment_score DESC 
    LIMIT 5
    """
    pos_df = pd.read_sql(query_pos, engine)
    for _, row in pos_df.iterrows():
        print(f"[+{row['roberta_sentiment_score']:.3f}] ({row['publish_date']}) {row['headline']}")

    # 3. Show 5 Most Negative Headlines
    print("\n--- Top 5 Most NEGATIVE Headlines (-1.0) ---")
    query_neg = """
    SELECT publish_date, roberta_sentiment_score, headline 
    FROM news_corpus 
    WHERE roberta_sentiment_score IS NOT NULL 
    ORDER BY roberta_sentiment_score ASC 
    LIMIT 5
    """
    neg_df = pd.read_sql(query_neg, engine)
    for _, row in neg_df.iterrows():
        print(f"[{row['roberta_sentiment_score']:.3f}] ({row['publish_date']}) {row['headline']}")
        
    print("\n==================================================")
    print("Verification Complete.")

if __name__ == "__main__":
    main()
