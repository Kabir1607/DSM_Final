# Project: Karnataka Education Policy Analysis (NEP 2020 vs. SEP 2025)

## 1. Project Overview
This project evaluates the causal impact of the 2020 National Education Policy (NEP) on higher education and job placement outcomes in Karnataka, compares it against public sentiment, and uses these findings to critique the proposed 2025 State Education Policy (SEP). 

The project culminates in an agentic LLM interface capable of executing text-to-SQL queries on tabular placement data and performing Retrieval-Augmented Generation (RAG) on government policy documents.

## 2. Core Objectives
* **Policy Metric Mapping:** Utilize the extracted specific higher education directives and metrics from government PDFs to guide quantitative analysis.
* **Causal Inference:** Execute a Difference-in-Differences (DiD) regression to prove the causal impact of the NEP on placements, directly testing hypotheses derived from existing academic literature.
    * *Treatment Group:* Karnataka (Early NEP adopter).
    * *Control Group:* Tamil Nadu or Kerala (Delayed/Resisted NEP).
    * *Time Periods:* Pre-Policy (2017–2019) vs. Post-Policy (2022–2025). Exclude 2020–2021 (COVID-19 shock).
* **Sentiment Overlay:** Perform sentiment analysis on historical newspaper headlines and text regarding the Karnataka NEP to compare institutional and public narrative trendlines against empirical outcomes.
* **Agentic Interface:** Build an interactive chat interface to query the database and policy text natively.

## 3. Tech Stack & Environment
* **Language:** Python 3.10+
* **Database:** PostgreSQL (Ideal for mixed tabular/JSONB text data) or SQLite.
* **Data Processing:** `pandas`, `numpy`, `sqlalchemy`
* **Causal Inference:** `statsmodels` (for OLS and DiD fixed effects)
* **NLP & Sentiment:** `transformers` (HuggingFace RoBERTa) for newspaper text sentiment.
* **LLM Integration:** `google-genai` (Gemini API for RAG and text-to-SQL routing)
* **Visualization/UI:** `matplotlib`, `seaborn`, `streamlit` or `gradio` 

## 4. Data Sources
1.  **NIRF Placement Data (2017-2025):** Kaggle dataset. Proxy for job placements and higher-ed transitions.
2.  **PLFS (Periodic Labour Force Survey):** MoSPI microdata. State-level employment indicators and macroeconomic controls.
3.  **AISHE (All India Survey on Higher Education):** NITI Aayog NDAP API. Gross Enrolment Ratio and institutional infrastructure.
4.  **Policy & Research Documents:** Local PDFs (NEP 2020, Karnataka Implementation, Karnataka SEP) and supporting academic literature.
5.  **API Guidelines:** `Data Ingestion for NEP Policy Analysis.md` contains the guidelines on how to develop the APIs to acquire this data.
6.  **Target Metrics:** `Analyzing Indian Higher Education Policy - Analyzing Indian Higher Education Policy.csv` contains the exact metrics to analyze.

## 5. Methodology & Execution Pipeline

### Phase 1: Policy Metric Mapping (Completed)
1.  Review the policy documents (NEP, Karnataka NEP Implementation, SEP Proposal).
2.  Extract the exact directives, proxy metrics, and target variables intended for higher education.
3.  These extracted parameters are saved in the `Analyzing Indian Higher Education Policy - Analyzing Indian Higher Education Policy.csv` file to serve as the variable roadmap for the DiD models.

### Phase 2: Data Acquisition & Metadata Searching (Current Focus)
1.  **Data Gathering:** Write Python scripts in the `src/data_gathering/` folder following the guidelines in the `Data Ingestion` markdown file to hit the NDAP, Kaggle, and GDELT APIs.
2.  **Explore Metadata:** Focus first on querying API endpoints to retrieve metadata, allowing you to search and see exactly what datasets and columns are available for Karnataka and the Control State.
3.  **Storage:** Download and save all gathered tabular and text datasets strictly into the `data/` folder.
4.  **Identify Confounders ($X_{ist}$):** Ensure the database schema tags institutions or districts as rural or urban. Create a rural/urban dummy variable to control for the massive structural disparities and lack of infrastructure highlighted in the literature.
5.  Clean and normalize the data, then build the relational schema.
6.  Perform Exploratory Data Analysis (EDA). Visualize pre-2020 and post-2020 distributions to verify the **Parallel Trends Assumption**.

### Phase 3: Causal Inference & Newspaper Sentiment Modeling
1.  **DiD Regression:**
    $$Y_{ist} = \alpha + \beta (Treatment_{s} \times Post2020_{t}) + \gamma_{s} + \delta_{t} + X_{ist}\theta + \epsilon_{ist}$$
2.  **Testable Hypotheses:** Use the regression to explicitly validate or refute claims from the existing literature:
    * *The Enrollment Drop:* Make Gross Enrolment Ratio (GER) a primary dependent variable ($Y$) using AISHE data to see if the NEP caused a negative impact.
    * *The Unemployment Spike:* Isolate the unemployment rate in the PLFS dataset to test if the policy raised educational unemployment.
    * *The Tech Placement Gap:* Segment NIRF placement data to compare technical vs. non-technical institutes to see if tech placements dropped due to failures in handling AI/cyberspace education.
3.  **Refined Sentiment Analysis:** * Scrape regional newspaper headlines and text (e.g., via GDELT or BeautifulSoup).
    * Filter data using specific pain points from the literature: *"Issues with examinations"*, *"workload of the teachers"*, and *"mismatch between skill and knowledge"*.
    * Run RoBERTa to track "Transition Resistance" over time and overlay this trendline on the $\beta$ outcome graphs.

### Phase 4: Synthesis & Agentic UI
1.  **Formulate SEP Recommendations:** Address the "Implementation Gap" highlighted in the literature. Use the empirical findings to suggest specific resource allocation strategies for the 2025 SEP that prevent multidisciplinary programs from disproportionately benefiting privileged students and widening the inequality gap.
2.  **Build the Agentic Router:**
    * *Intent = Data:* LLM writes SQL -> Queries DB -> Returns table/plot.
    * *Intent = Policy/Text:* LLM embeds query -> Vector search on PDF/TXT chunks -> Returns RAG response.

## 6. Directory Structure
```text
final_project/
├── .venv/
├── data/
├── Project_Requirements/
│   └── DSM Project - Guidelines and Expectations.pdf
├── Research_Documents/
│   ├── Policy_Documents/
│   │   ├── Karnataka_NEP.pdf
│   │   ├── Karnataka_Plan_SEP.pdf
│   │   └── NEP_National_Document.pdf
│   └── Secondary_Research/
│       ├── Data Ingestion for NEP Policy Analysis.md
│       └── Research_Paper_1.pdf
├── src/
│   ├── data_gathering/
│   ├── visualisations/
│   └── Analyzing Indian Higher Education Policy - Analyzing Indian Higher Education Policy.csv
├── .gitignore
├── context.md
├── Project_Report.txt
└── requirements.txt