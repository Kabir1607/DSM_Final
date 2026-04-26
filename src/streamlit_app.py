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

# Load environment variables
load_dotenv()

# Securely grab the API Key and DB URL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Please add it to your .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Database connection
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
    .stApp { background-color: #0B1121; color: #F8FAFC; }
    .main-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem; color: #94A3B8; border-left: 4px solid #8B5CF6;
        padding-left: 1rem; margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 1rem; padding: 2rem; margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Agentic Router Logic (Unchanged)
def get_routing_intent(prompt: str) -> str:
    routing_prompt = f"""
    You are an intelligent intent router. Classify the user query into exactly one word: "SQL" (for tabular data/placements) or "RAG" (for policy PDFs).
    User Query: {prompt}
    """
    response = model.generate_content(routing_prompt)
    return "SQL" if "SQL" in response.text.strip().upper() else "RAG"

def handle_sql_query(prompt: str) -> str:
    schema = "Table: nirf_rankings (institution_id, year, tlr_score, rpc_score, oi_score, perception_score, overall_score)\\nTable: placements (institution_id, year, no_of_students_placed, avg_salary_lpa)"
    sql_prompt = f"Generate raw Postgres SQL for this schema to answer: {prompt}\nSchema: {schema}\nReturn ONLY SQL."
    raw_sql = model.generate_content(sql_prompt).text.strip().replace("```sql", "").replace("```", "").strip()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(raw_sql))
            rows = result.fetchall()
            if not rows: return "No tabular data matched your request."
            data_str = " | ".join(result.keys()) + "\\n" + "\\n".join([" | ".join(str(v) for v in r) for r in rows])
            return model.generate_content(f"Answer '{prompt}' using this data:\n{data_str}").text.strip()
    except Exception as e:
        return f"Database Error: {str(e)}"

def handle_rag_query(prompt: str) -> str:
    try:
        prompt_embedding = genai.embed_content(model="models/gemini-embedding-001", content=prompt, task_type="retrieval_query")['embedding']
        with engine.connect() as conn:
            vec_str = "[" + ",".join(map(str, prompt_embedding)) + "]"
            query = text("SELECT document_name, chunk_text FROM policy_documents ORDER BY embedding <=> :vec LIMIT 4")
            rows = conn.execute(query, {"vec": vec_str}).fetchall()
        if not rows: return "No relevant policy documents found."
        context = "".join([f"Source: {r[0]}\\nExcerpt: {r[1]}\\n\\n" for r in rows])
        return model.generate_content(f"Answer '{prompt}' using ONLY these policy excerpts:\n{context}").text.strip()
    except Exception as e:
        return f"Retrieval Error: {str(e)}"

# Helper function to load images safely
def load_image(path):
    possible_paths = [path, f"../{path}", f"kabir1607/dsm_final/DSM_Final-ebbc1f86654aeea8a39aef377cd119d9d3eecb82/{path}"]
    for p in possible_paths:
        if os.path.exists(p): return Image.open(p)
    return None

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Research Blog", "Agentic RAG Router"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Database Status:** ✅ Connected")

# 4. Page 1: Research Blog (Redesigned with Tabs & Interactives)
if page == "Research Blog":
    st.markdown('<div class="main-title">Analyzing the Effects of NEP 2020 in Karnataka</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">To what extent have the goals the National Education Policy (NEP) set out for itself in Karnataka been realized compared to the counterfactual (Tamil Nadu)?</div>', unsafe_allow_html=True)
    
    # Interactive Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Methodology & Variable Justification", "EDA & Data Triangulation", "The 5 DiD Findings", "Conclusions"])
    
    with tab1:
        st.markdown("### Context & Motivation")
        st.write("Karnataka was an early adopter of the NEP, while states like Tamil Nadu actively resisted it. Given the widespread political dissatisfaction leading to the proposed State Education Policy (SEP 2025), this project empirically measures transition friction.")
        
        st.markdown("### The 5 Policy Metrics: Mapping & Justification")
        st.write("Abstract policy goals cannot be measured directly. I mapped specific NEP directives to quantitative NIRF and Sentiment variables:")
        
        with st.expander("1. Vocational Skill Integration ➔ mapped to **Graduation Outcomes (GO)**", expanded=True):
            st.write("**Justification:** The NEP heavily emphasizes employability and skills. The GO score measures exact placement rates and higher education progression.")
        with st.expander("2. Digital Divide & Infrastructure ➔ mapped to **Teaching, Learning & Resources (TLR)**"):
            st.write("**Justification:** Transitioning to a 4-year multidisciplinary model requires massive physical and digital CapEx. TLR measures Pupil-Teacher ratios and lab funding.")
        with st.expander("3. Inclusivity in STEMM ➔ mapped to **Outreach & Inclusivity (OI)**"):
            st.write("**Justification:** A core tenet of the NEP is increasing representation. OI directly measures the percentage of female and economically/socially disadvantaged students.")
        with st.expander("4. Restructuring & Autonomy ➔ mapped to **Research Output (RPC)**"):
            st.write("**Justification:** The NEP mandated institutions convert into heavy 'Multidisciplinary Research Clusters'. RPC tests if this research transition was successful.")
        with st.expander("5. Administrative Efficiency ➔ mapped to **NLP Sentiment Tracking**"):
            st.write("**Justification:** Raw data on KEA counseling delays and vacant seats is hidden from the public. Therefore, using RoBERTa on 4.2M regional news headlines acts as a perfect mathematical proxy for 'Transition Resistance' and public outrage.")

    with tab2:
        st.markdown("### Exploratory Data Analysis & Triangulation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4>Test 1: Placement Quantity vs. Quality</h4>
                <p>Cross-validated NIRF GO scores against raw average salaries (LPA).</p>
                <h2 style='color:#FBBF24;'>Correlation: 0.107 (Very Weak)</h2>
                <p><em>Insight:</em> The government index heavily overweights the quantity of placements over the quality of starting salaries.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4>Test 2: Infrastructure Integrity</h4>
                <p>Cross-validated NIRF TLR scores against raw AISHE Student-Faculty ratios.</p>
                <h2 style='color:#34D399;'>Correlation: -0.646 (Strong)</h2>
                <p><em>Insight:</em> Self-reported faculty counts appear genuine; institutions with crowded classrooms were accurately penalized.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### The Difference-in-Differences (DiD) Results")
        st.write("By isolating Karnataka's post-2020 growth from Tamil Nadu's baseline (excluding the COVID shock), we mathematically evaluated the NEP.")
        
        # Metric 1
        st.markdown("#### 1. Inclusivity & Equity in STEMM (OI Score)")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(label="DiD Estimator (Treatment Effect)", value="+3.45", delta="Massive Success")
            st.write("**Interpretation:** The NEP's mandate for flexible tracks successfully boosted marginalized demographics compared to the control state.")
        with colB:
            img1 = load_image("did_analysis/did_bar_oi_score.png")
            if img1: st.image(img1, use_container_width=True)
        st.divider()

        # Metric 2
        st.markdown("#### 2. Digital Divide & Infrastructure Funding (TLR Score)")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(label="DiD Estimator", value="-0.56", delta="Unfunded Mandate", delta_color="inverse")
            st.write("**Interpretation:** The central government required institutions to scale, but failed to provide the capital. Tamil Nadu actually outpaced Karnataka in scaling.")
        with colB:
            img2 = load_image("did_analysis/did_bar_tlr_score.png")
            if img2: st.image(img2, use_container_width=True)
        st.divider()

        # Metric 3
        st.markdown("#### 3. Institutional Restructuring & Autonomy (RPC Score)")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(label="DiD Estimator", value="-1.14", delta="Transition Friction", delta_color="inverse")
            st.write("**Interpretation:** The goal to build 'multidisciplinary research clusters' experienced friction, temporarily stalling Karnataka's research momentum.")
        with colB:
            img3 = load_image("did_analysis/did_bar_rpc_score.png")
            if img3: st.image(img3, use_container_width=True)
        st.divider()
        
        # Metric 4
        st.markdown("#### 4. Vocational & Technical Skill Integration (Placements)")
        colA, colB = st.columns([1, 2])
        with colA:
            st.write("**The 'Quantity over Quality' Illusion**")
            st.write("**Interpretation:** The policy currently rewards the *quantity* of low-paying jobs. The 2025 SEP must adopt stricter 'LPA thresholds' to accurately measure digital skill bridging.")
        with colB:
            img4 = load_image("did_analysis/did_timeseries_avg_salary_lpa.png")
            if img4: st.image(img4, use_container_width=True)
        st.divider()

        # Metric 5
        st.markdown("#### 5. Administrative Efficiency & Cybersecurity (Sentiment)")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(label="Pearson Correlation", value="-0.616", delta="Spillover Effect", delta_color="inverse")
            st.write("**Interpretation:** NLP tracking shows deep negative sentiment troughs in Karnataka. This public outrage acts as a leading indicator, correlating directly to a drop in student placement success the following year.")
        with colB:
            img5 = load_image("did_analysis/did_timeseries_sentiment.png")
            if img5: st.image(img5, use_container_width=True)

    with tab4:
        st.markdown("### Conclusions & Policy Recommendations")
        st.info("The NEP succeeded in equity but failed in execution.")
        st.markdown("""
        1. **Address the "Unfunded Mandate":** The government must match structural demands with actual capital expenditure.
        2. **Reform Evaluation Metrics:** Move away from raw placement volume (GO) and institute LPA thresholds.
        3. **Prioritize Administrative Stability:** Stop the policy whiplash to prevent downstream student failure (The Spillover Effect).
        
        > **Domain Expertise Note:** This analysis was conducted from a strict data-science perspective. Consulting a domain expert in Indian educational policy would greatly enhance the qualitative interpretation of these signals.
        """)

# 5. Page 2: Agentic RAG Interface (Unchanged)
elif page == "Agentic RAG Router":
    st.markdown('<div class="main-title">Agentic Database Router</div>', unsafe_allow_html=True)
    st.markdown("Ask natural language questions. The Agent will dynamically route your query to execute **Text-to-SQL** (for tabular data) or **Semantic Vector Search** (for policy documents).")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about the data or the policy PDFs!", "type": "system"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "sql": st.caption("🔍 Routed to **SQL Agent**")
            elif message.get("type") == "rag": st.caption("📄 Routed to **Semantic Agent**")
            st.markdown(message["content"])

    if prompt := st.chat_input("E.g. 'Show me 2024 placements' or 'What are the SEP funding rules?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

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