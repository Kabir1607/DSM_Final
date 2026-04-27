"""
013_rag_server.py
Optimized FastAPI backend for the Agentic Database Router.
Features: JSON-enforced routing, SQL self-healing, connection pooling, 
and Gemini Context Caching for whole-document RAG.
"""

import os
import json
import urllib.parse
import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import google.generativeai as genai
from google.generativeai import caching
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 1. Local Database Connection (For SQL queries)
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"

# Removed sslmode=require since localhost does not need external SSL
engine = create_engine(
    DB_URL, 
    pool_size=10, 
    max_overflow=20, 
    pool_pre_ping=True
)

# 2. Global Variables for Document Caching
PDF_DIR = "/home/Kdixter/Desktop/DSM_Final_Project/Research_Documents/Policy_Documents"
cached_rag_model = None
active_cache_name = None

# 3. Server Lifespan: Initialize Cache on Startup, Clean on Shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cached_rag_model, active_cache_name
    print("=" * 60)
    print("Initializing Gemini Context Cache with Policy PDFs...")
    
    uploaded_files = []
    if os.path.exists(PDF_DIR):
        for filename in os.listdir(PDF_DIR):
            if filename.endswith(".pdf"):
                filepath = os.path.join(PDF_DIR, filename)
                print(f"  -> Uploading {filename} to Gemini File API...")
                f = genai.upload_file(path=filepath, display_name=filename)
                uploaded_files.append(f)
    
    if uploaded_files:
        print(f"  -> Creating Context Cache for {len(uploaded_files)} documents...")
        system_instruction = (
            "You are an expert policy analyst. Answer the user's question using ONLY the provided "
            "official policy documents. Cite the source filename when appropriate. If the answer "
            "is not in the documents, state 'I don't have enough information in the policy documents to answer this.'"
        )
        
        # Note: Context caching requires specific versioned models. 
        # Using 1.5-flash-002 as it is the standard supported model for the caching API.
        cache = caching.CachedContent.create(
            model='models/gemini-1.5-flash-002',
            system_instruction=system_instruction,
            contents=uploaded_files,
            ttl=datetime.timedelta(hours=2) # Cache expires after 2 hours of inactivity
        )
        active_cache_name = cache.name
        
        # Instantiate a model bound directly to the cached documents
        cached_rag_model = genai.GenerativeModel.from_cached_content(cached_content=cache)
        print(f"✅ Cache active! Name: {active_cache_name}")
    else:
        print("❌ WARNING: No PDFs found in directory. RAG queries will fail.")
    print("=" * 60)
    
    yield # Server runs here
    
    # Teardown logic
    if active_cache_name:
        print(f"\nCleaning up Gemini Cache ({active_cache_name})...")
        caching.CachedContent.get(active_cache_name).delete()
        print("Cleanup complete.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Standard Model Initialization (For Routing and SQL)
model = genai.GenerativeModel('gemini-2.5-flash')

def get_routing_intent(prompt: str) -> str:
    """Uses JSON schema enforcement for lightning-fast, deterministic routing."""
    routing_prompt = f"""
    Classify the following query into exactly one category: "SQL" (tabular data like rankings, salaries, placements) or "RAG" (policy documents, NEP 2020 guidelines, text rules).
    Query: "{prompt}"
    """
    
    response = model.generate_content(
        routing_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": ["SQL", "RAG"]}
                },
                "required": ["intent"]
            }
        )
    )
    
    try:
        result = json.loads(response.text)
        return result.get("intent", "RAG")
    except Exception:
        return "RAG" 

def handle_sql_query(prompt: str, retry_count=0) -> str:
    """Translates NL to SQL, executes it, and features a self-reflection loop for errors."""
    schema = """
    Table: nirf_rankings (institution_id, year, tlr_score, rpc_score, oi_score, perception_score, overall_score, rank)
    Table: placements (institution_id, year, no_of_students_placed, median_salary, avg_salary_lpa)
    Table: institutions (institution_id, name, state, university_type)
    """
    
    sql_prompt = f"""
    You are a Postgres SQL expert. Generate a valid SQL query to answer the user's question.
    Schema: {schema}
    Return ONLY the raw SQL query. No markdown, no explanations.
    Question: {prompt}
    """
    
    raw_sql = model.generate_content(sql_prompt).text.replace("```sql", "").replace("```", "").strip()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(raw_sql))
            rows = result.fetchall()
            columns = result.keys()
            
            if not rows:
                return "I executed the query but found no tabular data matching your request."
            
            data_str = " | ".join(columns) + "\n"
            for row in rows:
                data_str += " | ".join(str(val) for val in row) + "\n"
                
            synth_prompt = f"""
            The user asked: "{prompt}"
            Database returned:
            {data_str}
            Write a clear, professional response answering the question using this data.
            """
            return model.generate_content(synth_prompt).text.strip()
            
    except Exception as e:
        if retry_count < 1:
            correction_prompt = f"The query I asked for: '{prompt}'. My previous SQL: {raw_sql}. It resulted in this error: {str(e)}. Please provide a corrected SQL query returning ONLY the raw SQL."
            return handle_sql_query(correction_prompt, retry_count=1)
        
        return f"I attempted to query the database but encountered an SQL error: {str(e)}"

def handle_rag_query(prompt: str) -> str:
    """Queries the cached model directly. NO pgvector similarity search needed!"""
    if not cached_rag_model:
        return "The document cache was not initialized. Please check the server logs."
        
    try:
        # The prompt goes straight to the model holding the 205 pages in memory
        response = cached_rag_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I encountered an error querying the document cache: {str(e)}"

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")
    
    if not user_prompt:
        return {"response": "Please provide a prompt.", "type": "system"}
        
    intent = get_routing_intent(user_prompt)
    
    if intent == "SQL":
        answer = handle_sql_query(user_prompt)
    else:
        answer = handle_rag_query(user_prompt)
        
    return {"response": answer, "type": intent.lower()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)