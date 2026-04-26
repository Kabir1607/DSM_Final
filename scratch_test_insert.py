import urllib.parse
from sqlalchemy import create_engine, text

encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)

try:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO policy_documents (document_name, chunk_text, embedding)
            VALUES ('test', 'test chunk', '[0.1, 0.2]')
        """))
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
