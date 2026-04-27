import os
import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables for the DB credentials
load_dotenv()

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

engine = create_engine(DB_URL)

def get_go_score_counts_by_state():
    """Fetches the years and the count of available GO scores for specific states."""
    
    # We use GROUP BY to aggregate the data, and COUNT() to find the volume per year
    query = text("""
        SELECT i.state, n.year, COUNT(n.go_score) as valid_scores
        FROM nirf_rankings n
        JOIN institutions i ON n.institution_id = i.institution_id
        WHERE n.go_score IS NOT NULL 
          AND i.state IN ('Tamil Nadu', 'Karnataka')
        GROUP BY i.state, n.year
        ORDER BY i.state, n.year;
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            
            # Group the returned (year, count) tuples by state
            state_data = defaultdict(list)
            for row in result.fetchall():
                state = row[0]
                year = row[1]
                count = row[2]
                state_data[state].append((year, count))
                
            # Format and print the results
            print("=" * 60)
            for state, data in state_data.items():
                print(f"📊 GO Score Volume for {state}:")
                for year, count in data:
                    print(f"   -> Year {year}: {count} valid scores")
                print("-" * 30)
            print("=" * 60)
            
            return dict(state_data)
            
    except Exception as e:
        print(f"❌ Database error encountered: {str(e)}")
        
if __name__ == "__main__":
    get_go_score_counts_by_state()