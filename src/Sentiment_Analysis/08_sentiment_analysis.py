import os
import re
import urllib.parse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, select, update, Table, MetaData
from sqlalchemy.orm import sessionmaker

# =====================================================================
# Database Configuration
# =====================================================================
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

# Create Engine with Connection Pooling
engine = create_engine(
    DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800
)

# =====================================================================
# Constants & Keywords
# =====================================================================
CATEGORY_A_POLICY = [r"\bnep\b", r"national education policy", r"\bsep\b", r"\bugc\b", r"higher education"]
CATEGORY_B_GEOG = ["karnataka", "tamil nadu", "bangalore", "chennai", "bengaluru"]

def _compile_regex(word_list):
    """Compiles a list of string patterns into a single case-insensitive regex."""
    return re.compile("|".join(word_list), re.IGNORECASE)

RE_POLICY = _compile_regex(CATEGORY_A_POLICY)
RE_GEOG = _compile_regex(CATEGORY_B_GEOG)

# =====================================================================
# Step 1: Data Extraction & Filtering
# =====================================================================
def extract_and_filter_corpus(engine):
    """
    Extracts all headlines from the database, applies regex filtering 
    based on Policy and Geography categories, and returns a DataFrame.
    """
    print("[1] Extracting and Filtering Headlines from DB...")
    
    # We query all headlines that haven't been scored yet
    query = """
    SELECT article_id, publish_date, headline 
    FROM news_corpus 
    WHERE roberta_sentiment_score IS NULL
    """
    
    df = pd.read_sql(query, con=engine)
    total_unscored = len(df)
    print(f"    - Pulled {total_unscored:,} unscored rows from database.")
    
    # Filter A: Must contain Policy keyword
    mask_policy = df['headline'].str.contains(RE_POLICY, na=False)
    # Filter B: Must contain Geography keyword
    mask_geog = df['headline'].str.contains(RE_GEOG, na=False)
    
    filtered_df = df[mask_policy & mask_geog].copy()
    print(f"    - Filtered down to {len(filtered_df):,} target 'Education Policy' rows.")
    
    return filtered_df

# =====================================================================
# Step 2: RoBERTa Sentiment Scoring
# =====================================================================
def map_to_polarity(probs):
    """
    Maps the [Negative, Neutral, Positive] probability vector from 
    cardiffnlp/twitter-roberta-base-sentiment-latest to a [-1.0, 1.0] scale.
    """
    # Weights for Neg, Neu, Pos
    weights = torch.tensor([-1.0, 0.0, 1.0], device=probs.device)
    # Calculate expected value (continuous score)
    scores = (probs * weights).sum(dim=1)
    return scores.cpu().numpy()

def score_sentiment(df, batch_size=64):
    """
    Scores the sentiment of headlines using a HuggingFace RoBERTa model.
    Runs on CUDA or MPS if available.
    """
    if len(df) == 0:
        print("    - No rows to score. Skipping.")
        return df
        
    print("[2] Initializing RoBERTa Model...")
    
    # Determine best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("    - Device: CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("    - Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("    - Device: CPU (Will be slow!)")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    print(f"    - Scoring {len(df):,} headlines in batches of {batch_size}...")
    
    headlines = df['headline'].tolist()
    scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(headlines), batch_size), desc="Scoring"):
            batch_texts = headlines[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_scores = map_to_polarity(probs)
            scores.extend(batch_scores)
            
    df['roberta_sentiment_score'] = scores
    # Round to 3 decimal places as per schema NUMERIC(4,3)
    df['roberta_sentiment_score'] = df['roberta_sentiment_score'].round(3)
    return df

# =====================================================================
# Step 3: Database Update
# =====================================================================
def update_database_scores(engine, scored_df):
    """
    Updates the news_corpus table in PostgreSQL with the new sentiment scores
    using batched, safe SQLAlchemy transactions.
    """
    if len(scored_df) == 0:
        return
        
    print(f"[3] Updating {len(scored_df):,} rows in the Database...")
    
    metadata = MetaData()
    news_table = Table('news_corpus', metadata, autoload_with=engine)
    
    # Prepare list of dictionaries for update
    update_data = [
        {
            'b_article_id': row['article_id'],
            'b_score': row['roberta_sentiment_score']
        }
        for _, row in scored_df.iterrows()
    ]
    
    Session = sessionmaker(bind=engine)
    
    # Execute batch updates
    with Session() as session:
        # We use a simple loop over the batched records to update each
        for i in tqdm(range(0, len(update_data), 1000), desc="Updating DB"):
            batch = update_data[i:i+1000]
            for record in batch:
                upd = update(news_table).where(news_table.c.article_id == record['b_article_id']).values(roberta_sentiment_score=record['b_score'])
                session.execute(upd)
        
        session.commit()
    print("    - Database updated successfully!")

# =====================================================================
# Step 4: Time-Series Aggregation
# =====================================================================
def generate_timeseries_aggregation(engine):
    """
    Queries the database for scored headlines, categorizes them by state, 
    and calculates a 3-month rolling average sentiment.
    Returns a Pandas DataFrame.
    """
    print("[4] Generating Time-Series Aggregations...")
    
    query = """
    SELECT publish_date, headline, roberta_sentiment_score 
    FROM news_corpus 
    WHERE roberta_sentiment_score IS NOT NULL
    """
    
    df = pd.read_sql(query, con=engine)
    
    if len(df) == 0:
        print("    - No scored data found in the database. Returning empty DataFrame.")
        return df
        
    # Extract State based on Geography Keywords
    def extract_state(text):
        text = text.lower()
        if "karnataka" in text or "bangalore" in text or "bengaluru" in text:
            return "Karnataka"
        elif "tamil nadu" in text or "chennai" in text:
            return "Tamil Nadu"
        return "Unknown"
        
    df['state'] = df['headline'].apply(extract_state)
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['year_month'] = df['publish_date'].dt.to_period('M')
    
    # Group by State and Year-Month to get monthly averages
    monthly_avg = df.groupby(['state', 'year_month'])['roberta_sentiment_score'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values(by=['state', 'year_month'])
    
    # Calculate 3-Month Rolling Average per State
    monthly_avg['rolling_3m_sentiment'] = monthly_avg.groupby('state')['roberta_sentiment_score'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    print("    - Time-series aggregation complete.")
    return monthly_avg

# =====================================================================
# Main Execution
# =====================================================================
def main():
    print("==========================================================")
    print("  NEP Sentiment Analysis Pipeline (RoBERTa)")
    print("==========================================================")
    
    # Step 1: Extract & Filter
    filtered_df = extract_and_filter_corpus(engine)
    
    # Step 2: Score Sentiment
    scored_df = score_sentiment(filtered_df, batch_size=64)
    
    # Step 3: Update DB
    update_database_scores(engine, scored_df)
    
    # Step 4: Generate Aggregations (for plotting in Jupyter later)
    ts_df = generate_timeseries_aggregation(engine)
    
    # Save the aggregated output for easy plotting
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sentiment_timeseries_aggregation.csv")
    if not ts_df.empty:
        ts_df.to_csv(out_path, index=False)
        print(f"    - Saved aggregated metrics to: {out_path}")

if __name__ == "__main__":
    main()
