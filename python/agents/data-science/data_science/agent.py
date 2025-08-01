# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top level agent for data agent multi-agents.

-- it get data from database (e.g., BQ) using NL2SQL
-- then, it use NL2Py to do further data analysis as needed
"""
import os
from datetime import date

from google.genai import types

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_artifacts

from .sub_agents import bqml_agent
from .sub_agents.bigquery.tools import (
    get_database_settings as get_bq_database_settings,
)
from .prompts import return_instructions_root
from .tools import call_db_agent, call_ds_agent

# Import PostgreSQL support for Eric's academic corpus
try:
    from .academic_database_agent import get_academic_agent
    ERIC_CORPUS_AVAILABLE = True
except ImportError:
    ERIC_CORPUS_AVAILABLE = False

date_today = date.today()


def get_postgresql_database_settings():
    """Get PostgreSQL database settings for Eric's academic corpus."""
    schema_description = """
    Eric's Academic Corpus Schema (PostgreSQL):
    
    -- Main documents table
    CREATE TABLE papers (
        paper_id VARCHAR PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT[], -- Array of author names
        publication_year INTEGER,
        doi VARCHAR,
        abstract TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Page-accurate document chunks with dual embeddings
    CREATE TABLE paper_chunks (
        chunk_id VARCHAR PRIMARY KEY,
        paper_id VARCHAR REFERENCES papers(paper_id),
        paper_title TEXT,
        title TEXT, -- Section/chunk title
        content TEXT NOT NULL,
        section_number VARCHAR,
        section_title TEXT,
        chunk_type VARCHAR, -- introduction, methodology, conclusion, etc.
        page_numbers INTEGER[], -- Pages this chunk spans
        word_count INTEGER,
        academic_quality_score DECIMAL(3,2), -- CHASE-SQL quality score 0.0-1.0
        specter2_embedding VECTOR(768), -- Academic similarity embedding
        openai_embedding VECTOR(3072), -- General semantic embedding
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Additional chunk metadata
    CREATE TABLE chunk_metadata (
        chunk_id VARCHAR REFERENCES paper_chunks(chunk_id),
        keywords TEXT[], -- Extracted keywords
        key_concepts TEXT[], -- Key conceptual terms
        citations TEXT[], -- Referenced citations in chunk
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Sample data examples:
    -- papers: 111 academic papers on Frankfurt School critical theory
    -- paper_chunks: 31,847 concept-based chunks with page-accurate references
    -- Academic focus: dialectical materialism, critical theory, social philosophy
    -- Chunk types: introduction, methodology, analysis, conclusion, references
    -- Page accuracy: All chunks maintain precise page number tracking
    -- Quality scores: CHASE-SQL academic compliance scoring (0.0-1.0)
    -- Dual embeddings: SPECTER2 for academic similarity, OpenAI for general search
    """
    
    return {
        "pg_schema": schema_description,
        "corpus_type": "eric_academic",
        "total_papers": 111,
        "total_chunks": 31847,
        "embedding_models": ["specter2", "openai"],
        "academic_focus": "Frankfurt School critical theory",
    }


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the agent."""

    # Check if Eric's corpus mode is enabled
    use_eric_corpus = os.getenv("USE_ERIC_CORPUS", "false").lower() == "true"
    
    # setting up database settings in session.state
    if "database_settings" not in callback_context.state:
        db_settings = dict()
        if use_eric_corpus and ERIC_CORPUS_AVAILABLE:
            db_settings["use_database"] = "PostgreSQL"
        else:
            db_settings["use_database"] = "BigQuery"
        callback_context.state["all_db_settings"] = db_settings

    # setting up schema in instruction
    if callback_context.state["all_db_settings"]["use_database"] == "PostgreSQL":
        callback_context.state["database_settings"] = get_postgresql_database_settings()
        schema = callback_context.state["database_settings"]["pg_schema"]
        
        callback_context._invocation_context.agent.instruction = (
            return_instructions_root()
            + f"""

    --------- Eric's Academic Corpus Schema (PostgreSQL) with 111 papers and 31,847 concept-based chunks ---------
    {schema}
    
    Academic Research Context:
    - This database contains Eric's corpus of Frankfurt School critical theory papers
    - Use CHASE-SQL multi-path reasoning for complex academic queries
    - All chunks include page-accurate references for proper academic citations
    - Dual embedding system: SPECTER2 for academic similarity, OpenAI for general search
    - Academic quality scores (0.0-1.0) indicate scholarly compliance and relevance

    """
        )
    elif callback_context.state["all_db_settings"]["use_database"] == "BigQuery":
        callback_context.state["database_settings"] = get_bq_database_settings()
        schema = callback_context.state["database_settings"]["bq_ddl_schema"]

        callback_context._invocation_context.agent.instruction = (
            return_instructions_root()
            + f"""

    --------- The BigQuery schema of the relevant data with a few sample rows. ---------
    {schema}

    """
        )


root_agent = Agent(
    model=os.getenv("ROOT_AGENT_MODEL"),
    name="db_ds_multiagent",
    instruction=return_instructions_root(),
    global_instruction=(
        f"""
        You are a Data Science and Data Analytics Multi Agent System.
        Todays date: {date_today}
        """
    ),
    sub_agents=[bqml_agent],
    tools=[
        call_db_agent,
        call_ds_agent,
        load_artifacts,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
