import os
import fitz # PyMuPDF
import google.generativeai as genai
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import urllib.parse
import time

# Load ENV from the correct location
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

if "GEMINI_API_KEY" not in os.environ:
    print("ERROR: GEMINI_API_KEY not found in environment")
    exit(1)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PDF_DIR = "/home/Kdixter/Desktop/DSM_Final_Project/Research_Documents/Policy_Documents"
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)

def setup_db():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS policy_documents (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255),
                chunk_index INTEGER,
                content TEXT,
                embedding vector(768)
            );
        """))

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def extract_and_chunk():
    chunks_data = []
    
    if not os.path.exists(PDF_DIR):
        print(f"Error: Directory {PDF_DIR} does not exist.")
        return []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith('.pdf'): continue
        filepath = os.path.join(PDF_DIR, filename)
        
        print(f"Reading {filename}...")
        try:
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
                
            chunks = chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                chunks_data.append((filename, i, chunk))
        except Exception as e:
            print(f"Failed reading {filename}: {e}")
            
    return chunks_data

def embed_and_store(chunks_data):
    if not chunks_data:
        return

    # Check if we already stored
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM policy_documents")).scalar()
        if count > 0:
            print(f"Database already has {count} chunks. Dropping existing chunks for a fresh run...")
            with engine.begin() as drop_conn:
                drop_conn.execute(text("TRUNCATE TABLE policy_documents"))

    print(f"Embedding {len(chunks_data)} chunks...")
    
    model = "models/gemini-embedding-001"
    
    with engine.connect() as conn:
        for i, (source, c_idx, content) in enumerate(chunks_data):
            if i % 10 == 0: 
                print(f"Processed {i}/{len(chunks_data)} chunks...")
            
            time.sleep(4) # Guarantee < 15 requests per minute for free tier
                
            try:
                response = genai.embed_content(
                    model=model,
                    content=content,
                    task_type="retrieval_document"
                )
                embedding = response['embedding']
                emb_str = "[" + ",".join(map(str, embedding)) + "]"
                
                conn.execute(text("""
                    INSERT INTO policy_documents (document_name, chunk_text, embedding)
                    VALUES (:s, :c, :e)
                """), {"s": source, "c": content, "e": emb_str})
                conn.commit()
            except Exception as e:
                print(f"Error on chunk {i}: {e}")
                conn.rollback()
                time.sleep(10)

if __name__ == "__main__":
    setup_db()
    data = extract_and_chunk()
    embed_and_store(data)
    print("Ingestion complete.")
