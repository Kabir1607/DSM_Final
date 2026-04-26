"""
013_rag_server.py
FastAPI backend for the Agentic Database Router.
Connects the React frontend to Gemini and PostgreSQL (tabular & vector).
"""
import os
import urllib.parse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Setup Database
encoded_password = urllib.parse.quote_plus("School#1607")
DB_URL = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
engine = create_engine(DB_URL)

app = FastAPI()

# Allow CORS for the static frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini Model
model = genai.GenerativeModel('gemini-2.5-flash')

def get_routing_intent(prompt: str) -> str:
    """Uses an LLM to determine if the query is asking about tabular data or policy documents."""
    routing_prompt = f"""
    You are an intelligent intent router for a database system.
    The database contains two types of data:
    1. SQL: Tabular data about universities (NIRF rankings, TLR scores, placement outcomes, average salaries in LPA).
    2. RAG: Unstructured text data from government policy PDFs (NEP 2020, SEP 2025 proposals, funding rules, guidelines).

    Classify the following user query. Respond with EXACTLY ONE WORD: "SQL" or "RAG".
    
    User Query: {prompt}
    """
    response = model.generate_content(routing_prompt)
    intent = response.text.strip().upper()
    return "SQL" if "SQL" in intent else "RAG"

def handle_sql_query(prompt: str) -> str:
    """Translates natural language to SQL, executes it, and synthesizes the answer."""
    schema = """
    Table: nirf_rankings (institution_id, year, tlr_score, rpc_score, oi_score, perception_score, overall_score, rank)
    Table: placements (institution_id, year, no_of_students_placed, median_salary, avg_salary_lpa)
    Table: institutions (institution_id, name, state, university_type)
    """
    
    sql_prompt = f"""
    You are a Postgres SQL expert. Generate a valid SQL query to answer the user's question based on this schema:
    {schema}
    
    Return ONLY the raw SQL query. No markdown formatting, no explanation. Do not include ```sql.
    Make sure to limit results to 5 rows if appropriate.
    
    User Question: {prompt}
    """
    
    sql_response = model.generate_content(sql_prompt)
    raw_sql = sql_response.text.strip()
    # Clean up formatting if Gemini added it
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(raw_sql))
            rows = result.fetchall()
            columns = result.keys()
            
            if not rows:
                return f"I executed the query but found no tabular data matching your request."
            
            # Format the data for the LLM to synthesize
            data_str = " | ".join(columns) + "\\n"
            for row in rows:
                data_str += " | ".join(str(val) for val in row) + "\\n"
                
            synth_prompt = f"""
            The user asked: "{prompt}"
            I ran a SQL query and got these results:
            {data_str}
            
            Write a clear, concise, professional response directly answering the user's question using this data.
            """
            final_response = model.generate_content(synth_prompt)
            return final_response.text.strip()
            
    except Exception as e:
        return f"I attempted to query the database but encountered an SQL error: {str(e)}\n\nQuery generated was: {raw_sql}"

def handle_rag_query(prompt: str) -> str:
    """Embeds the prompt, queries pgvector, and synthesizes the text chunks."""
    try:
        # Get vector for the prompt
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=prompt,
            task_type="retrieval_query"
        )
        prompt_embedding = response['embedding']
        
        # Execute vector similarity search in pgvector using Cosine Distance (<=>)
        with engine.connect() as conn:
            # Format embedding array for Postgres syntax
            vector_str = "[" + ",".join(map(str, prompt_embedding)) + "]"
            query = text("""
                SELECT document_name, chunk_text, 1 - (embedding <=> :vec) as similarity
                FROM policy_documents
                ORDER BY embedding <=> :vec
                LIMIT 4
            """)
            result = conn.execute(query, {"vec": vector_str})
            rows = result.fetchall()
            
        if not rows:
            return "I couldn't find any relevant policy documents for your query."
            
        # Compile context
        context = ""
        for r in rows:
            context += f"Source: {r[0]}\\nExcerpt: {r[1]}\\n\\n"
            
        synth_prompt = f"""
        You are an expert policy analyst. Answer the user's question using ONLY the provided excerpts from official policy documents.
        If the excerpts do not contain the answer, say "I don't have enough information in the policy documents to answer this."
        Cite the source filename when appropriate.
        
        User Question: {prompt}
        
        Policy Excerpts:
        {context}
        """
        
        final_response = model.generate_content(synth_prompt)
        return final_response.text.strip()
        
    except Exception as e:
        return f"I encountered an error during semantic vector retrieval: {str(e)}"

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")
    
    if not user_prompt:
        return {"response": "Please provide a prompt.", "type": "system"}
        
    # 1. Routing
    intent = get_routing_intent(user_prompt)
    
    # 2. Execution
    if intent == "SQL":
        answer = handle_sql_query(user_prompt)
    else:
        answer = handle_rag_query(user_prompt)
        
    return {"response": answer, "type": intent.lower()}

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
