import os
import urllib.parse
import streamlit as st
import google.generativeai as genai
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from PIL import Image

# 1. Configuration & Setup
st.set_page_config(
    page_title="NEP Policy Analysis Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables (Local .env or Streamlit Secrets)
load_dotenv()

# Securely grab the API Key and DB URL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Please add it to your .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Database connection - Easily swappable to Supabase
# To switch to supabase, just change the DB_URL in your .env or Streamlit secrets
encoded_password = urllib.parse.quote_plus("School#1607")
default_local_db = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
DB_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL", default_local_db)

@st.cache_resource
def init_connection():
    return create_engine(DB_URL)

engine = init_connection()

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Premium Glassmorphism & Colors */
    .stApp {
        background-color: #0B1121;
        color: #F8FAFC;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #94A3B8;
        border-left: 4px solid #8B5CF6;
        padding-left: 1rem;
        margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 1rem;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34D399 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-warning {
        background: linear-gradient(135deg, #F87171 0%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Agentic Router Logic
def get_routing_intent(prompt: str) -> str:
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
    schema = """
    Table: nirf_rankings (institution_id, year, tlr_score, rpc_score, oi_score, perception_score, overall_score, rank)
    Table: placements (institution_id, year, no_of_students_placed, median_salary, avg_salary_lpa)
    Table: institutions (institution_id, name, state, university_type)
    """
    
    sql_prompt = f"""
    You are a Postgres SQL expert. Generate a valid SQL query to answer the user's question based on this schema:
    {schema}
    
    Return ONLY the raw SQL query. No markdown formatting, no explanation. Do not include ```sql.
    Limit results to 5 rows if appropriate.
    
    User Question: {prompt}
    """
    
    sql_response = model.generate_content(sql_prompt)
    raw_sql = sql_response.text.strip().replace("```sql", "").replace("```", "").strip()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(raw_sql))
            rows = result.fetchall()
            columns = result.keys()
            
            if not rows:
                return f"I executed the query but found no tabular data matching your request."
            
            data_str = " | ".join(columns) + "\\n"
            for row in rows:
                data_str += " | ".join(str(val) for val in row) + "\\n"
                
            synth_prompt = f"""
            The user asked: "{prompt}"
            I ran a SQL query and got these results:
            {data_str}
            Write a concise, professional response answering the user's question using this data.
            """
            final_response = model.generate_content(synth_prompt)
            return final_response.text.strip()
            
    except Exception as e:
        return f"Database SQL Error: {str(e)}\n\nGenerated Query: {raw_sql}"

def handle_rag_query(prompt: str) -> str:
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=prompt,
            task_type="retrieval_query"
        )
        prompt_embedding = response['embedding']
        
        with engine.connect() as conn:
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
        return f"Semantic Retrieval Error: {str(e)}"

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Research Blog", "Agentic RAG Router"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Database Status:** ✅ Connected")
st.sidebar.markdown(f"**Target:** `{'Supabase' if 'supabase' in DB_URL else 'Local PostgreSQL'}`")

# Helper function to load images safely
def load_image(path):
    # Try relative path first, then absolute path if running locally
    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    if os.path.exists(full_path):
        return Image.open(full_path)
    # Fallback to absolute local path just in case
    abs_path = os.path.join("/home/Kdixter/Desktop/DSM_Final_Project", path)
    if os.path.exists(abs_path):
        return Image.open(abs_path)
    return None

# 4. Page 1: Research Blog
if page == "Research Blog":
    st.markdown('<div class="main-title">Analyzing the Effects of NEP 2020 in Karnataka</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Research Question: To what extent have the goals the National Education Policy (NEP) set out for itself in Karnataka been realized compared to the counterfactual?</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3>Context & Motivation</h3>
        <p><strong>Secondary Research:</strong> Grounded in the official NEP 2020 document, the Karnataka implementation guidelines, and the recently proposed State Education Policy (SEP 2025).</p>
        <p><strong>Motivation:</strong> Karnataka was an early adopter of the NEP, while states like Tamil Nadu actively resisted it. This project empirically measures how well the policy's structural changes were actually realized using Difference-in-Differences econometrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Test 1: Placement Quantity vs. Quality**")
        st.info("Cross-validated NIRF GO scores against raw average salaries (LPA). Found a weak correlation of **0.107**.")
        img1 = load_image("final_project/data/processed/EDA_Reports/data_triangulation_results/test1_go_vs_lpa.png")
        if img1: st.image(img1, use_container_width=True)
    
    with col2:
        st.markdown("**Test 2: Infrastructure Integrity**")
        st.success("Cross-validated NIRF TLR scores against raw AISHE Student-Faculty ratios (PTR). Found a strong negative correlation of **-0.646**.")
        img2 = load_image("final_project/data/processed/EDA_Reports/data_triangulation_results/test2_tlr_vs_ptr.png")
        if img2: st.image(img2, use_container_width=True)

    st.markdown("---")
    st.header("Results: The DiD & Sentiment Findings")
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white;">1. The Inclusivity Success</h3>
        <p class="metric-value">DiD: +3.45 (OI Score)</p>
        <p><strong>Interpretation:</strong> The NEP's mandate for flexible tracks was a massive success for marginalized demographics, significantly boosting female and SC/ST/OBC enrollment in Karnataka compared to the control.</p>
    </div>
    """, unsafe_allow_html=True)
    img3 = load_image("did_analysis/did_bar_oi_score.png")
    if img3: st.image(img3, use_container_width=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white;">2. The Unfunded Mandate</h3>
        <p><span class="metric-value metric-warning">DiD: -0.56</span> (TLR) &nbsp;&nbsp;|&nbsp;&nbsp; <span class="metric-value metric-warning">DiD: -1.14</span> (RPC)</p>
        <p><strong>Interpretation:</strong> The policy mandated massive structural scaling but provided inadequate capital. Tamil Nadu outpaced Karnataka in infrastructure and research growth without the policy.</p>
    </div>
    """, unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    img4 = load_image("did_analysis/did_bar_tlr_score.png")
    img5 = load_image("did_analysis/did_bar_rpc_score.png")
    with col3:
        if img4: st.image(img4, use_container_width=True)
    with col4:
        if img5: st.image(img5, use_container_width=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white;">3. The Cost of Chaos</h3>
        <p><strong>r = -0.616</strong> (Lagged Outrage vs GO)</p>
        <p><strong>Interpretation:</strong> Administrative policy whiplash directly and negatively impacts student success. Public outrage serves as a leading indicator for institutional failure.</p>
    </div>
    """, unsafe_allow_html=True)
    img6 = load_image("did_analysis/did_timeseries_sentiment.png")
    if img6: st.image(img6, use_container_width=True)
    
    st.markdown("---")
    st.header("Conclusions & Recommendations")
    st.markdown("""
    - **Address the "Unfunded Mandate":** Match structural demands with actual capital expenditure.
    - **Reform Evaluation Metrics:** Prioritize high-paying job outcomes rather than raw placement volume.
    - **Prioritize Administrative Stability:** Stop the whiplash to prevent downstream student failure.
    
    > **Domain Expertise Note:** I approached this project strictly from a data-science perspective without prior knowledge of the Indian Education System. Consulting a dedicated domain expert in local educational policy would have greatly enhanced the qualitative interpretation of these signals.
    """)

# 5. Page 2: Agentic RAG Interface
elif page == "Agentic RAG Router":
    st.markdown('<div class="main-title">Agentic Database Router</div>', unsafe_allow_html=True)
    st.markdown("Ask natural language questions. The Agent will dynamically route your query to execute **Text-to-SQL** (for placements/rankings) or **Semantic Vector Search** (for policy documents).")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the Agentic Router. Ask me about the data or the policy PDFs!", "type": "system"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "sql":
                st.caption("🔍 Routed to **SQL Agent**")
            elif message.get("type") == "rag":
                st.caption("📄 Routed to **Semantic Agent**")
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("E.g. 'Show me 2024 placements' or 'What are the SEP funding rules?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent and routing..."):
                intent = get_routing_intent(prompt)
                
                if intent == "SQL":
                    st.caption("🔍 Routed to **SQL Agent**")
                    response = handle_sql_query(prompt)
                else:
                    st.caption("📄 Routed to **Semantic Agent**")
                    response = handle_rag_query(prompt)
                
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response, "type": intent.lower()})
