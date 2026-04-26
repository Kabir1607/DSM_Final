import urllib.parse
from sqlalchemy import create_engine, text

# 1. Store the raw password
raw_password = "School#1607"

# 2. URL-encode it
encoded_password = urllib.parse.quote_plus(raw_password)

# 3. Inject it into the connection string
db_url = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"))
    print("Extension Status:", result.fetchone())