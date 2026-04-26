import urllib.parse
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, 
    Numeric, Date, Text, UniqueConstraint, Index, text, ForeignKey, Float
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

# Initialize the declarative base
Base = declarative_base()

# ==========================================
# 1. DIMENSION TABLES (Static/Slow-Changing)
# ==========================================

class Institution(Base):
    """Master institution dimension table.
    
    Populated from:
      - NIRF Master CSV  → Institute ID, Institute Name, City, State
      - india_colleges.csv → name, city, state, type
      - AISHE college_institution.csv → id, name, state_code, autonomous, location, year_of_establishment
      - top_engineering_2025 → College Name, State, Type, Year Established
    """
    __tablename__ = 'institutions'
    
    institution_id = Column(Integer, primary_key=True, autoincrement=True)
    nirf_id = Column(String(50), nullable=True)           # e.g. "NIRF-ENGG-INF-77" from NIRF Master
    aishe_id = Column(Integer, nullable=True)              # numeric id from AISHE college_institution.csv
    name = Column(String(500), nullable=False)
    city = Column(String(100))
    state = Column(String(100), nullable=False)
    institution_type = Column(String(100))                 # IIT, NIT, State Univ, Private, etc.
    is_autonomous = Column(Boolean)                        # from AISHE 'autonomous' column
    is_rural = Column(Boolean)                             # from AISHE 'location' column
    year_established = Column(Integer)                     # from top_engineering_2025 or AISHE
    
    # Relationships
    rankings = relationship("NirfRanking", back_populates="institution")
    placements = relationship("Placement", back_populates="institution")
    aishe_enrollments = relationship("AisheEnrollment", back_populates="institution")
    aishe_infrastructure = relationship("AisheInfrastructure", back_populates="institution")

    __table_args__ = (
        UniqueConstraint('name', 'state', name='uq_institution_name_state'),
        Index('idx_inst_state', 'state'),
        Index('idx_inst_nirf_id', 'nirf_id'),
        Index('idx_inst_aishe_id', 'aishe_id'),
    )


class StateReference(Base):
    """State lookup table from AISHE ref_state.csv.
    
    Populated from:
      - AISHE ref_state.csv → st_code, name
    """
    __tablename__ = 'state_reference'

    st_code = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)


# ==========================================
# 2. FACT TABLES (Quantitative / DiD Models)
# ==========================================

class NirfRanking(Base):
    """NIRF institutional rankings per year with sub-scores.
    
    Populated from:
      - data/processed/NIRF_Master_2016_2025.csv
        Columns: Institute ID, Institute Name, TLR, RPC, GO, OI, PERCEPTION,
                 City, State, Score, Rank, Year
      - data/kaggle/nirf/nirf_2024_detailed/NIRF Ranking 2024.csv
        Adds: Field (Engineering, Overall, etc.)
    """
    __tablename__ = 'nirf_rankings'
    
    ranking_id = Column(Integer, primary_key=True, autoincrement=True)
    institution_id = Column(Integer, ForeignKey('institutions.institution_id'))
    year = Column(Integer, nullable=False)
    rank = Column(Integer)
    overall_score = Column(Numeric(5, 2))
    tlr_score = Column(Numeric(5, 2))       # Teaching, Learning & Resources
    rpc_score = Column(Numeric(5, 2))       # Research & Professional Practice
    go_score = Column(Numeric(5, 2))        # Graduation Outcomes
    oi_score = Column(Numeric(5, 2))        # Outreach & Inclusivity
    perception_score = Column(Numeric(5, 2))
    field = Column(String(100))              # Engineering, Overall, etc. (from 2024 detailed)

    institution = relationship("Institution", back_populates="rankings")

    __table_args__ = (
        UniqueConstraint('institution_id', 'year', 'field', name='uq_nirf_inst_year_field'),
        Index('idx_nirf_year', 'year'),
        Index('idx_nirf_state_year', 'institution_id', 'year'),
    )


class Placement(Base):
    """Placement outcome data for institutions.
    
    Populated from:
      - india_colleges.csv → placement_avg_lpa, fees_ug_inr, rating, nirf_rank
      - top_engineering_2025 → Average Placement %, Highest Package (LPA),
                                Average Package (LPA), Student-Faculty Ratio
      - college_placement_general/detailed_data.csv → company, college, region, year, Salary
    """
    __tablename__ = 'placements'

    placement_id = Column(Integer, primary_key=True, autoincrement=True)
    institution_id = Column(Integer, ForeignKey('institutions.institution_id'), nullable=True)
    year = Column(Integer, nullable=True)
    company_name = Column(String(255), nullable=True)
    avg_salary_lpa = Column(Float, nullable=True)
    highest_package_lpa = Column(Float, nullable=True)
    avg_placement_pct = Column(Float, nullable=True)
    fees_ug_inr = Column(Float, nullable=True)
    student_faculty_ratio = Column(Float, nullable=True)
    source = Column(String(100))            # 'india_colleges', 'top_engineering_2025', 'college_placement_general'

    institution = relationship("Institution", back_populates="placements")

    __table_args__ = (
        Index('idx_placement_year', 'year'),
        Index('idx_placement_institution', 'institution_id'),
    )


class AisheEnrollment(Base):
    """AISHE enrollment counts by course, category, and year.
    
    Populated from:
      - AISHE enrolled_student_count.csv (per survey year directory)
        Columns: id, course_mode_id, level_id, programme_id, discipline,
                 course_type_id, year, count_by_category_id, broad_discipline_group_id
      - Cross-referenced with ref_state.csv via college_institution.state_code
    
    Note: The raw AISHE data uses numeric foreign keys (course_mode_id, level_id, etc.)
          that reference separate ref_*.csv lookup tables.
    """
    __tablename__ = 'aishe_enrollment'

    enrollment_id = Column(Integer, primary_key=True, autoincrement=True)
    institution_id = Column(Integer, ForeignKey('institutions.institution_id'), nullable=True)
    aishe_record_id = Column(Integer)       # original 'id' from CSV
    survey_year = Column(String(20), nullable=False)  # e.g. '2015_16'
    course_mode_id = Column(Integer)         # Regular / Distance
    level_id = Column(Integer)               # UG / PG / PhD
    programme_id = Column(Integer)           # BA, BSc, BTech, etc.
    discipline = Column(Integer)             # Subject discipline code
    course_type_id = Column(Integer)         # Certificate / Diploma / Degree
    enrollment_count = Column(Integer)       # from count_by_category_id join
    broad_discipline_group_id = Column(Integer)  # STEM, Arts, etc.

    institution = relationship("Institution", back_populates="aishe_enrollments")

    __table_args__ = (
        Index('idx_aishe_enroll_year', 'survey_year'),
        Index('idx_aishe_enroll_institution', 'institution_id'),
    )


class AisheInfrastructure(Base):
    """AISHE infrastructure data per institution.
    
    Populated from:
      - AISHE infrastructure.csv (per survey year directory)
        Key columns: id, playground, library, laboratory, computer_center,
                     no_of_books, no_of_journals, connectivity_nkn, connectivity_nmeict,
                     solar_power_generation, campus_friendly
    """
    __tablename__ = 'aishe_infrastructure'

    infra_id = Column(Integer, primary_key=True, autoincrement=True)
    institution_id = Column(Integer, ForeignKey('institutions.institution_id'), nullable=True)
    aishe_record_id = Column(Integer)       # original 'id' from CSV
    survey_year = Column(String(20), nullable=False)
    has_library = Column(Boolean)
    has_laboratory = Column(Boolean)
    has_computer_center = Column(Boolean)
    has_playground = Column(Boolean)
    no_of_books = Column(Integer)
    no_of_journals = Column(Integer)
    no_of_computer_centers = Column(Integer)
    no_of_laboratories = Column(Integer)
    connectivity_nkn = Column(Boolean)       # National Knowledge Network
    connectivity_nmeict = Column(Boolean)    # NMEICT connectivity
    solar_power = Column(Boolean)
    campus_friendly = Column(Boolean)        # Disability-accessible

    institution = relationship("Institution", back_populates="aishe_infrastructure")

    __table_args__ = (
        Index('idx_aishe_infra_year', 'survey_year'),
        Index('idx_aishe_infra_institution', 'institution_id'),
    )


class MacroControl(Base):
    """State-level macroeconomic control variables for DiD.
    
    Populated from:
      - data/kaggle/employment/unemployment_india/Unemployment in India.csv
        Columns: Region, Date, Frequency, Estimated Unemployment Rate (%),
                 Estimated Employed, Estimated Labour Participation Rate (%), Area
    """
    __tablename__ = 'macro_controls'
    
    control_id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(100), nullable=False)
    date = Column(Date, nullable=False)
    area = Column(String(20))               # 'Rural' or 'Urban'
    estimated_unemployment_rate = Column(Numeric(5, 2))
    estimated_employed = Column(Numeric(15, 2))
    estimated_lfpr = Column(Numeric(5, 2))  # Labour Force Participation Rate

    __table_args__ = (
        UniqueConstraint('state', 'date', 'area', name='uq_macro_state_date_area'),
        Index('idx_macro_state', 'state'),
        Index('idx_macro_date', 'date'),
    )

# ==========================================
# 3. UNSTRUCTURED TABLES (Text, RAG & NLP)
# ==========================================

class PolicyDocument(Base):
    """Chunked policy documents for RAG retrieval.
    
    Populated from:
      - Manual ingestion of NEP 2020 PDF text, UGC circulars, etc.
      - Not directly from any downloaded CSV (prepared manually)
    """
    __tablename__ = 'policy_documents'
    
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String(255), nullable=False)
    section_heading = Column(String(255))
    chunk_text = Column(Text, nullable=False)
    
    # Nullable by default: Ready for NLP later
    embedding = Column(Vector(768), nullable=True) 

    __table_args__ = (
        # HNSW Index for Vector Similarity Search
        Index('idx_policy_embedding', 'embedding', 
              postgresql_using='hnsw', 
              postgresql_with={'m': 16, 'ef_construction': 64}, 
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )


class NewsCorpus(Base):
    """News headlines and articles for sentiment analysis.
    
    Populated from:
      - data/kaggle/news/india_news_headlines/india-news-headlines.csv
        Columns: publish_date (YYYYMMDD int), headline_category, headline_text
        Rows: 3,876,557 | Span: 2001-2023
      - data/kaggle/news/et_headlines_2022_2025/economic_times_headlines_*.csv
        Columns: Archive, Date (DD-MM-YYYY), Headline, Headline link
        Rows: ~350K | Span: 2022-2025
      - data/kaggle/news/financial_news_2003_2020/IndianFinancialNews.csv
        Columns: Date (text), Title, Description
        Rows: 50,000 | Span: 2003-2020
      - GDELT API articles (78 articles from exploration)
      - NewsAPI articles (48 articles from exploration)
    """
    __tablename__ = 'news_corpus'
    
    article_id = Column(Integer, primary_key=True, autoincrement=True)
    publish_date = Column(Date, nullable=False)
    source_name = Column(String(100))       # 'india_headlines', 'economic_times', 'financial_news', 'gdelt', 'newsapi'
    category = Column(String(100))          # headline_category from india-news-headlines.csv
    headline = Column(Text, nullable=False)
    description = Column(Text)              # article body/description (financial_news, newsapi)
    url = Column(Text)                      # article URL (ET, financial_news, newsapi)

    # Nullable by default: Ready for NLP later
    roberta_sentiment_score = Column(Numeric(4, 3), nullable=True) 
    embedding = Column(Vector(768), nullable=True)                 

    __table_args__ = (
        UniqueConstraint('headline', 'publish_date', 'source_name', name='uq_news_headline_date_source'),
        
        # HNSW Index for semantic RAG searches
        Index('idx_news_embedding', 'embedding', 
              postgresql_using='hnsw', 
              postgresql_with={'m': 16, 'ef_construction': 64}, 
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
              
        # GIN Index for fast keyword matching without LLM usage
        Index('idx_news_headline', text("to_tsvector('english', headline)"), postgresql_using='gin'),

        # B-Tree for time-based filtering (DiD pre/post windows)
        Index('idx_news_date', 'publish_date'),
        Index('idx_news_source', 'source_name'),
    )

# ==========================================
# DATABASE INITIALIZATION
# ==========================================

if __name__ == "__main__":
    # Safely URL-encode the password with the '#' character
    encoded_password = urllib.parse.quote_plus("School#1607")
    db_url = f"postgresql://nep_admin:{encoded_password}@localhost:5432/nep_db"
    
    engine = create_engine(db_url, echo=True) # echo=True prints the SQL execution

    with engine.connect() as conn:
        # Ensure the vector extension is created before building tables
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    # Generate all tables and indexes mapped in the classes above
    Base.metadata.create_all(engine)
    
    print("\n✅ Database schema successfully deployed to nep_db!")