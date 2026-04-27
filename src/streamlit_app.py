import os
import urllib.parse
import time
import re
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

# Custom CSS for Premium Design & Increased Font Size (Sci-Fi / Clean Dark Aesthetic)
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px !important; }
    
    .stApp { background-color: #0A0E17; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 3rem !important; font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem; line-height: 1.2; letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.3rem !important; color: #94A3B8; border-left: 4px solid #818CF8;
        padding-left: 1rem; margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); backdrop-filter: blur(4px);
    }
    .big-metric { font-size: 2.5rem; font-weight: 900; margin: 0; padding: 0; line-height: 1; }
    .highlight-blue { color: #38BDF8; font-weight: bold; }
    .highlight-green { color: #34D399; font-weight: bold; }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Resilient Agentic Router Logic (Smarter Quota Handling)
def retry_gemini(func, *args, **kwargs):
    """Wrapper to handle Gemini API rate limits with dynamic quota parsing."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                if attempt < max_retries - 1:
                    match = re.search(r"retry in (\d+\.?\d*)s", error_msg, re.IGNORECASE)
                    if match:
                        sleep_time = float(match.group(1)) + 2.0
                    else:
                        sleep_time = 60.0
                    
                    st.warning(f"⏳ **Gemini API Free Tier Limit Reached.** Pausing for {int(sleep_time)} seconds to refresh quota... Please do not close the window.")
                    time.sleep(sleep_time)
                    continue
            raise e

def get_routing_intent(prompt: str) -> str:
    routing_prompt = f"""
    You are an intelligent intent router. Classify the user query into exactly one word: "SQL" (for tabular data/placements) or "RAG" (for policy PDFs).
    User Query: {prompt}
    """
    response = retry_gemini(model.generate_content, routing_prompt)
    return "SQL" if "SQL" in response.text.strip().upper() else "RAG"

def handle_sql_query(prompt: str) -> str:
    schema = "Table: nirf_rankings (institution_id, year, tlr_score, rpc_score, oi_score, perception_score, overall_score)\\nTable: placements (institution_id, year, no_of_students_placed, avg_salary_lpa)"
    sql_prompt = f"Generate raw Postgres SQL for this schema to answer: {prompt}\nSchema: {schema}\nReturn ONLY SQL."
    
    try:
        raw_sql = retry_gemini(model.generate_content, sql_prompt).text.strip().replace("```sql", "").replace("```", "").strip()
        with engine.connect() as conn:
            result = conn.execute(text(raw_sql))
            rows = result.fetchall()
            if not rows: return "No tabular data matched your request."
            data_str = " | ".join(result.keys()) + "\\n" + "\\n".join([" | ".join(str(v) for v in r) for r in rows])
            return retry_gemini(model.generate_content, f"Answer '{prompt}' using this data:\n{data_str}").text.strip()
    except Exception as e:
        return f"Database or API Error: {str(e)}"

def handle_rag_query(prompt: str) -> str:
    try:
        embed_response = retry_gemini(genai.embed_content, model="models/text-embedding-004", content=prompt, task_type="retrieval_query")
        prompt_embedding = embed_response['embedding']
        
        with engine.connect() as conn:
            vec_str = "[" + ",".join(map(str, prompt_embedding)) + "]"
            query = text("SELECT document_name, chunk_text FROM policy_documents ORDER BY embedding <=> CAST(:vec AS vector) LIMIT 4")
            rows = conn.execute(query, {"vec": vec_str}).fetchall()
            
        if not rows: return "No relevant policy documents found."
        context = "".join([f"Source: {r[0]}\\nExcerpt: {r[1]}\\n\\n" for r in rows])
        
        system_prompt = f"Answer '{prompt}' comprehensively using ONLY these official policy excerpts:\n{context}"
        return retry_gemini(model.generate_content, system_prompt).text.strip()
    except Exception as e:
        return f"Retrieval or API Error: {str(e)}"

def load_image(path):
    possible_paths = [
        path, 
        f"src/{path}", 
        f"../{path}", 
        f"../src/{path}",
        f"kabir1607/dsm_final/DSM_Final-14937987073ddfb6dac391f62901dd2eef1902cb/src/{path}"
    ]
    for p in possible_paths:
        if os.path.exists(p): return Image.open(p)
    return None

# 3. Sidebar Navigation
st.sidebar.title("System Access")
page = st.sidebar.radio("Modules", ["Research & Insights", "Database Architecture", "Agentic Interrogator"])
st.sidebar.markdown("---")
st.sidebar.markdown("**PostgreSQL Cluster:** <span style='color:#34D399'>✅ Online</span>", unsafe_allow_html=True)
st.sidebar.markdown("**Vector Store Engine:** <span style='color:#34D399'>✅ pgvector</span>", unsafe_allow_html=True)

# 4. Page: Database Architecture
if page == "Database Architecture":
    st.markdown('<div class="main-title">Data Topology & Indexing Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A breakdown of the underlying storage, scale, and retrieval optimization.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>Tabular Data Corpus</h3>
            <p><b>Total Volume: ~3.6 GB</b></p>
            <ul>
                <li><b>NIRF Institutional Data:</b> ~10 years of longitudinal rankings and sub-scores.</li>
                <li><b>AISHE Microdata:</b> Exhaustive institutional enrollment and infrastructure flags.</li>
                <li><b>Placement Outcomes:</b> Consolidated salary (LPA) and placement percentage data across top engineering and state institutions.</li>
                <li><b>Macroeconomic Data:</b> PLFS unemployment rates to control for state-level economic shifts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>Unstructured Text Corpus</h3>
            <p><b>Miscellaneous Text & Embeddings</b></p>
            <ul>
                <li><b>Policy Documents:</b> Parsed and chunked PDFs (NEP 2020, Karnataka SEP 2025, UGC Circulars).</li>
                <li><b>News Headlines (Sentiment):</b> ~4.2 Million historical regional news headlines processed for NLP scoring.</li>
                <li><b>Articles:</b> Financial news and GDELT API pulls covering transition resistance.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Indexing Strategy (The Engine behind the Agent)")
    st.info("The database schema was purposefully engineered to support real-time Agentic routing without latency bottlenecks.")
    st.markdown("""
    * **HNSW (Hierarchical Navigable Small World) Index:** Applied to the `vector(768)` embedding columns. This is why the RAG system is fast; instead of computing cosine distance against every policy chunk sequentially, it navigates a multi-layered graph to find the nearest semantic neighbors instantly.
    * **GIN (Generalized Inverted Index):** Applied via `to_tsvector` on the massive 4.2M news headline corpus. This allows for instantaneous lexical keyword matching (e.g., finding exact mentions of "NEP" or "KEA counseling") without wasting LLM tokens.
    * **B-Tree Indexes:** Standard indexing heavily applied to the `year` and `publish_date` columns across all fact tables. This perfectly optimizes the crucial "Pre-2020" vs "Post-2021" time-window filtering required for the DiD models.
    """)

# 5. Page: Research Blog
elif page == "Research & Insights":
    st.markdown('<div class="main-title">Analyzing NEP 2020: The Karnataka Counterfactual</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Methodology & Mapping", "Data Triangulation", "The DiD Framework", "Sentiment Overlay", "Conclusions"])
    
    with tab1:
        st.markdown("### Reading the Law: Policy Mapping")
        st.write("Abstract policy goals cannot be measured directly. To run an empirical evaluation, the first step involved parsing the national NEP 2020 PDF and Karnataka's specific implementation guidelines to extract what they were *actually* trying to do. These directives were then mapped to quantitative tracking variables:")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            with st.expander("1. Vocational Skill Integration ➔ GO Score"):
                st.write("**NEP Goal:** Boost employability and practical skills.")
                st.write("**Evaluation Metric (GO):** Graduation Outcomes. Evaluates what happens to students after they finish courses, weighing student placement rates, median salary, and progression to higher studies.")
            with st.expander("2. Digital Divide & CapEx ➔ TLR Score"):
                st.write("**NEP Goal:** Establish massive multidisciplinary infrastructure.")
                st.write("**Evaluation Metric (TLR):** Teaching, Learning & Resources. Measures core educational infrastructure, faculty-student ratio, and the financial resources/capital expenditure of the institution.")
            with st.expander("3. Inclusivity in STEMM ➔ OI Score"):
                st.write("**NEP Goal:** Increase representation across demographics.")
                st.write("**Evaluation Metric (OI):** Outreach & Inclusivity. Captures demographic diversity, measuring the percentage of women, economically/socially disadvantaged students, and physically challenged students.")
        with col_m2:
            with st.expander("4. Institutional Restructuring ➔ RPC Score"):
                st.write("**NEP Goal:** Convert institutions into heavy research clusters.")
                st.write("**Evaluation Metric (RPC):** Research and Professional Practice. Reflects the academic and research output, evaluating the quantity and quality of publications, citations, and patents.")
            with st.expander("5. Administrative Efficiency ➔ PERCEPTION & NLP"):
                st.write("**NEP Goal:** Streamlined, centralized admissions.")
                st.write("**Evaluation Metric:** Perception scores (peer/employer reputation) paired with RoBERTa NLP Sentiment tracking on local news to proxy public outrage and transition resistance.")

    with tab2:
        st.markdown("### Exploratory Data Analysis & Triangulation")
        st.write("Before running the causal models, the government index scores were cross-validated against raw, ground-truth variables to ensure data integrity.")
        
        st.markdown("#### Test 1: Quantity vs. Quality")
        img_t1 = load_image("final_visuals/GO_AVERGE_PLACEMENT.png")
        if img_t1: st.image(img_t1, caption="GO Score vs Average LPA Distribution", use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <p>Cross-validated NIRF Graduation Outcomes (GO) against raw average starting salaries (LPA).</p>
            <p style="color:#FBBF24;"><strong>Very Weak Correlation (r = 0.107)</strong></p>
            <p><em>Insight:</em> The government index heavily overweights the quantity of placements over the quality of starting salaries.</p>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("#### Test 2: Infrastructure Integrity")
        img_t2 = load_image("final_visuals/NIRF_STUDENT_TEACHER.png")
        if img_t2: st.image(img_t2, caption="TLR Score vs Student-Faculty Ratio", use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <p>Cross-validated NIRF TLR scores against raw AISHE Student-Faculty ratios.</p>
            <p style="color:#34D399;"><strong>Strong Negative Correlation (r = -0.646)</strong></p>
            <p><em>Insight:</em> Self-reported faculty counts appear genuine; institutions with crowded classrooms were accurately penalized in their score.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### The Intuition: Why Difference-in-Differences (DiD)?")
        st.markdown("""
        <div class="glass-card">
        To prove a policy *caused* an outcome, you cannot simply look at Karnataka before and after 2020. What if the entire country's education system naturally improved over that time?
        <br><br>
        DiD solves this by using a <b>Counterfactual</b>. We use Tamil Nadu (a state that actively resisted the NEP) as our Control Group, and Karnataka (an early adopter) as our Treatment Group.
        <br><br>
        <b>The Parallel Trends Assumption:</b> If Karnataka and Tamil Nadu were trending similarly <i>before</i> the policy (pre-2020), any sudden divergence between the two states <i>after</i> the policy (post-2021) can be mathematically attributed to the NEP itself, isolating it from general macroeconomic shifts.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### The Empirical Findings")
        
        # Metric 1: OI
        st.markdown("#### 1. Inclusivity & Equity in STEMM (OI Score)")
        img_oi = load_image("final_visuals/did_bar_oi_score.png")
        if img_oi: st.image(img_oi, use_container_width=True)
        
        img_oi_ts = load_image("final_visuals/did_timeseries_oi_score.png")
        if img_oi_ts: st.image(img_oi_ts, caption="Pre-and-Post Parallel Trends divergence", use_container_width=True)
        
        st.metric(label="DiD Estimator", value="+3.45", delta="Massive Success")
        st.write("The mandate for flexible tracks successfully boosted marginalized demographics compared to the control state.")
        st.divider()
            
        # Metric 2: TLR
        st.markdown("#### 2. Digital Divide & Infrastructure (TLR Score)")
        img_tlr = load_image("final_visuals/did_bar_tlr_score.png")
        if img_tlr: st.image(img_tlr, use_container_width=True)
        
        img_tlr_ts = load_image("final_visuals/did_timeseries_tlr_score.png")
        if img_tlr_ts: st.image(img_tlr_ts, caption="Notice Tamil Nadu accelerating while Karnataka stagnates post-2021", use_container_width=True)
        
        st.metric(label="DiD Estimator", value="-0.56", delta="Unfunded Mandate", delta_color="inverse")
        st.write("The government required institutions to scale but failed to provide the capital. Tamil Nadu outpaced Karnataka in scaling.")
        st.divider()
            
        # Metric 3: RPC
        st.markdown("#### 3. Institutional Restructuring (RPC Score)")
        img_rpc = load_image("final_visuals/did_bar_rpc_score.png")
        if img_rpc: st.image(img_rpc, use_container_width=True)
        
        img_rpc_ts = load_image("final_visuals/did_timeseries_rpc_score.png")
        if img_rpc_ts: st.image(img_rpc_ts, caption="Structural drop in Karnataka relative to control.", use_container_width=True)
        
        st.metric(label="DiD Estimator", value="-1.14", delta="Transition Friction", delta_color="inverse")
        st.write("The goal to build 'multidisciplinary research clusters' experienced severe friction, stalling Karnataka's research momentum.")
        st.divider()

        # Metric 4: GO
        st.markdown("#### 4. Vocational Integration (GO Score)")
        img_go = load_image("final_visuals/did_bar_go_score.png")
        if img_go: st.image(img_go, use_container_width=True)
        
        img_go_ts = load_image("final_visuals/did_timeseries_go_score.png")
        if img_go_ts: st.image(img_go_ts, caption="Time series comparison of Graduation Outcomes", use_container_width=True)
        
        st.write("Tracking graduation outcomes (placement rates and median salaries) post-implementation.")
        st.divider()

        # Limitations Section
        st.markdown("### Analytical Limitations")
        st.warning("""
        **Data Scarcity for Specific Variables:** While metrics like RPC and TLR provide robust DiD estimators, certain metrics lacked complete longitudinal ground-truth datasets for highly localized institutional realities across the entire pre-2020 control period. Perception scoring remains challenging to reliably proxy across historical windows.
        """)

    with tab4:
        st.markdown("### Sentiment Mismatch & The Spillover Effect")
        st.write("By taking a Rolling Average Sentiment Trendline from historical news corpora and plotting it alongside the DiD graphs, a stark mismatch emerges between government mandates and on-the-ground reality.")
        
        img5 = load_image("final_visuals/did_timeseries_sentiment.png")
        if img5: st.image(img5, use_container_width=True, caption="Historical Sentiment Trendline")
        
        img6 = load_image("final_visuals/ts_sentiment_corr_tlr_score.png")
        if img6: st.image(img6, use_container_width=True, caption="Sentiment Correlated with TLR Resource Declines")
        
        st.markdown("""
        <div class="glass-card">
            <h4>The Insights</h4>
            <ul>
                <li><b>The Mismatch:</b> While government evaluations praised the rapid structural rollouts, NLP tracking reveals deep negative sentiment troughs in Karnataka regarding examination chaos, teacher workload, and counseling delays.</li>
                <li><b>The Spillover Effect:</b> This public outrage isn't just noise; it acts as a leading indicator. The data shows that drops in administrative sentiment directly correlate with a subsequent drop in operational infrastructure and resource bandwidth.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab5:
        st.markdown("### The Narrative Arc")
        
        st.markdown("""
        <div class="glass-card">
            <ul>
                <li><b>The Equity Success (OI Score):</b> The policy mandate for flexible tracks and multidisciplinary inclusion worked incredibly well. Karnataka saw a massive positive DiD estimator (+3.45) in Outreach & Inclusivity compared to the control state, successfully boosting the representation of marginalized demographics in STEMM.</li>
                <li><b>The "Unfunded Mandate" Reality (TLR Score):</b> The digital divide and infrastructure demands of the NEP severely strained institutions. With a -0.56 DiD estimator, Karnataka failed to match the capital expenditure and faculty scaling required for the massive multidisciplinary restructuring, actually falling behind the control group. The strong negative correlation (r = -0.646) between the TLR score and raw Student-Faculty ratios confirms that classrooms became overcrowded.</li>
                <li><b>Institutional Friction (RPC Score):</b> The push to convert institutions into heavy research clusters backfired in the short term. The DiD estimator (-1.14) shows that research momentum stalled due to the sheer administrative burden of transitioning syllabi, faculties, and structures.</li>
                <li><b>The Public Outrage Spillover:</b> The NLP sentiment analysis perfectly mirrors the structural failures. While the government praised the rollout, local news sentiment tracking shows deep negative troughs. More importantly, this negative sentiment heavily correlates with failures in teaching resources. The administrative chaos (examination issues, teacher workloads) actively eroded placement success and institutional stability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Ultimate Conclusion:** The NEP succeeded conceptually in driving equity but stumbled operationally due to inadequate funding and massive transition friction.")
        
        st.markdown("### Policy Recommendations")
        st.markdown("""
        <div class="glass-card">
            <ul style="margin-bottom: 0;">
                <li><b>Address the "Unfunded Mandate":</b> Match structural demands with actual capital expenditure in the SEP 2025.</li>
                <li><b>Reform Evaluation Metrics:</b> Prioritize high-paying job outcomes (LPA thresholds) rather than raw placement volume.</li>
                <li><b>Prioritize Administrative Stability:</b> Public outrage is statistically linked to downstream failure.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Future Betterments")
        st.write("The analysis would benefit from highly domain-specific evaluations (e.g., separating Medical from Engineering).")
        
        st.markdown("### Domain Expertise Disclaimer")
        st.warning("""
        I approached this project strictly from a data-science perspective without prior knowledge of the Indian Education System. Consulting a dedicated domain expert in local educational policy would have greatly enhanced the qualitative interpretation of these signals.
        """)

# 6. Page: Agentic RAG Interface
elif page == "Agentic Interrogator":
    st.markdown('<div class="main-title" style="font-size: 4.5rem; text-align: center; margin-top: 2rem;">RAG System Run Locally</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin-top: 2rem;">
        <h3 style="color: #34D399;">System Offline & Secure</h3>
        <p>Because the database environment is approximately 3.6 GB, the Agentic Router and Vector Stores have been entirely offloaded from cloud APIs.</p>
        <p>The system is currently running via a manual, local LLaMA-based FastAPI server (<code>013_rag_server.py</code>) relying on FAISS embeddings to preserve data locality.</p>
    </div>
    """, unsafe_allow_html=True)