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

"""PostgreSQL tools for Eric's academic corpus with CHASE-SQL integration."""

import os
import logging
import asyncio
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from google.adk.tools import ToolContext
from google.genai import Client

# Add Eric's corpus processor and academic database agent to path
sys.path.append('/home/scott/research/eric-corpus-processor/src')
sys.path.append('/home/scott/research/adk-samples/python/agents/data-science')

try:
    from data_science.academic_database_agent import get_academic_agent
    ACADEMIC_AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Academic agent not available: {e}")
    ACADEMIC_AGENT_AVAILABLE = False

# PostgreSQL connection configuration
PG_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'paper_chunks'),
    'user': os.getenv('POSTGRES_USER', 'eric_production'),
    'password': os.getenv('POSTGRES_PASSWORD', 'secure_production_2025')
}

MAX_NUM_ROWS = 50
logger = logging.getLogger(__name__)

# Initialize Vertex AI client for CHASE-SQL
vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT", None)
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
llm_client = Client(vertexai=True, project=vertex_project, location=location)


def get_postgresql_database_settings():
    """Get PostgreSQL database settings for Eric's corpus."""
    return {
        "pg_host": PG_CONFIG['host'],
        "pg_database": PG_CONFIG['database'],
        "pg_schema": get_academic_schema_description(),
        "corpus_type": "eric_academic",
        "total_papers": 111,
        "total_chunks": 31847,
    }


def get_academic_schema_description():
    """Get detailed academic corpus schema description."""
    return """
    Eric's Academic Corpus - Frankfurt School Critical Theory (PostgreSQL)
    
    Tables:
    1. papers (111 academic papers)
       - paper_id, title, authors[], publication_year, doi, abstract
       
    2. paper_chunks (31,847 concept-based chunks)
       - chunk_id, paper_id, paper_title, title, content
       - section_number, section_title, chunk_type
       - page_numbers[] (page-accurate references)
       - word_count, academic_quality_score (0.0-1.0)
       - specter2_embedding[768], openai_embedding[3072]
       
    3. chunk_metadata (additional academic metadata)
       - chunk_id, keywords[], key_concepts[], citations[]
       
    Academic Focus: dialectical materialism, critical theory, social philosophy
    Chunk Types: introduction, methodology, analysis, conclusion, references
    Quality Scoring: CHASE-SQL academic compliance (higher = more scholarly)
    """


def academic_nl2sql(question: str, tool_context: ToolContext) -> str:
    """
    Generate SQL query for Eric's academic corpus using CHASE-SQL methodology.
    
    Args:
        question: Natural language question about the academic corpus
        tool_context: ADK tool context
        
    Returns:
        PostgreSQL query string
    """
    logger.info(f"Generating academic SQL for: {question}")
    
    # CHASE-SQL inspired prompt for academic queries
    prompt_template = """
You are an expert in converting natural language questions into PostgreSQL queries for Eric's academic corpus of Frankfurt School critical theory papers.

**Database Schema:**
{SCHEMA}

**Guidelines for Academic Queries:**
- Use semantic search principles for concept-based queries
- Leverage page_numbers for citation accuracy
- Use academic_quality_score for scholarly relevance ranking
- Consider chunk_type for section-specific searches
- Join tables appropriately for comprehensive results
- Include page references in results for academic citations
- Limit results to {MAX_ROWS} rows maximum

**Academic Query Patterns:**
- Paper searches: SELECT from papers table with metadata
- Content searches: SELECT from paper_chunks with content analysis
- Citation searches: JOIN with chunk_metadata for citations
- Concept searches: Use key_concepts and keywords fields
- Quality filtering: ORDER BY academic_quality_score DESC

**Natural Language Question:**
{QUESTION}

Generate a PostgreSQL query that accurately answers this academic research question:
"""

    schema = get_academic_schema_description()
    
    prompt = prompt_template.format(
        SCHEMA=schema,
        MAX_ROWS=MAX_NUM_ROWS,
        QUESTION=question
    )
    
    try:
        response = llm_client.models.generate_content(
            model=os.getenv("BASELINE_NL2SQL_MODEL", "gemini-2.0-flash-exp"),
            contents=prompt,
            config={"temperature": 0.1},
        )
        
        sql = response.text
        if sql:
            sql = sql.replace("```sql", "").replace("```", "").strip()
        
        logger.info(f"Generated academic SQL: {sql}")
        tool_context.state["sql_query"] = sql
        
        return sql
        
    except Exception as e:
        logger.error(f"Failed to generate academic SQL: {e}")
        # Fallback to simple corpus overview query
        fallback_sql = """
        SELECT 
            COUNT(DISTINCT p.paper_id) as total_papers,
            COUNT(DISTINCT c.chunk_id) as total_chunks,
            ROUND(AVG(c.academic_quality_score), 3) as avg_quality_score,
            COUNT(DISTINCT c.chunk_type) as chunk_types
        FROM papers p
        LEFT JOIN paper_chunks c ON p.paper_id = c.paper_id
        """
        tool_context.state["sql_query"] = fallback_sql
        return fallback_sql


def run_postgresql_validation(sql_string: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Execute and validate PostgreSQL query for Eric's academic corpus.
    
    Args:
        sql_string: PostgreSQL query to execute
        tool_context: ADK tool context
        
    Returns:
        Dictionary with query results or error information
    """
    logger.info(f"Executing academic query: {sql_string}")
    
    final_result = {"query_result": None, "error_message": None}
    
    # Clean up the SQL
    sql_string = sql_string.strip()
    if not sql_string.upper().startswith('SELECT'):
        final_result["error_message"] = "Only SELECT queries are allowed for academic corpus"
        return final_result
    
    # Add LIMIT if not present
    if "limit" not in sql_string.lower():
        sql_string = sql_string.rstrip(';') + f" LIMIT {MAX_NUM_ROWS};"
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Execute query
        cursor.execute(sql_string)
        rows = cursor.fetchall()
        
        # Format results for ADK
        if rows:
            results = []
            for row in rows:
                formatted_row = {}
                for key, value in row.items():
                    # Handle PostgreSQL arrays and special types
                    if isinstance(value, list):
                        formatted_row[key] = value
                    elif isinstance(value, datetime):
                        formatted_row[key] = value.isoformat()
                    else:
                        formatted_row[key] = value
                results.append(formatted_row)
            
            final_result["query_result"] = results
            tool_context.state["query_result"] = results
            
            logger.info(f"Academic query returned {len(results)} rows")
        else:
            final_result["error_message"] = "Query executed successfully but returned no results"
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"PostgreSQL query failed: {e}")
        final_result["error_message"] = f"Academic query error: {str(e)}"
    
    return final_result


async def search_academic_content(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Search Eric's academic corpus using semantic similarity.
    
    Args:
        query: Search query
        tool_context: ADK tool context
        
    Returns:
        Dictionary with search results
    """
    if not ACADEMIC_AGENT_AVAILABLE:
        return {"error": "Academic search agent not available"}
    
    try:
        # Get the academic agent instance
        academic_agent = get_academic_agent()
        
        # Perform similarity search
        similar_chunks = await academic_agent.search_similar_content(
            query_text=query,
            embedding_type="openai",  # Use general semantic search
            limit=10
        )
        
        # Format results for ADK
        results = {
            "search_query": query,
            "results_count": len(similar_chunks),
            "similar_chunks": similar_chunks,
            "search_type": "semantic_similarity"
        }
        
        tool_context.state["search_results"] = results
        return results
        
    except Exception as e:
        logger.error(f"Academic content search failed: {e}")
        return {"error": f"Search failed: {str(e)}"}


async def get_corpus_overview(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get overview statistics of Eric's academic corpus.
    
    Args:
        tool_context: ADK tool context
        
    Returns:
        Dictionary with corpus overview
    """
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get comprehensive corpus statistics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT p.paper_id) as total_papers,
                COUNT(DISTINCT c.chunk_id) as total_chunks,
                ROUND(AVG(c.academic_quality_score), 3) as avg_quality_score,
                COUNT(DISTINCT c.chunk_type) as unique_chunk_types,
                SUM(c.word_count) as total_words,
                MIN(p.publication_year) as earliest_year,
                MAX(p.publication_year) as latest_year,
                COUNT(DISTINCT CASE WHEN c.specter2_embedding IS NOT NULL THEN c.chunk_id END) as specter2_embeddings,
                COUNT(DISTINCT CASE WHEN c.openai_embedding IS NOT NULL THEN c.chunk_id END) as openai_embeddings
            FROM papers p
            LEFT JOIN paper_chunks c ON p.paper_id = c.paper_id
        """)
        
        stats = cursor.fetchone()
        
        # Get sample chunk types
        cursor.execute("""
            SELECT chunk_type, COUNT(*) as count
            FROM paper_chunks
            WHERE chunk_type IS NOT NULL
            GROUP BY chunk_type
            ORDER BY count DESC
            LIMIT 10
        """)
        
        chunk_types = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        overview = {
            "corpus_name": "Eric's Frankfurt School Critical Theory Corpus",
            "statistics": dict(stats) if stats else {},
            "chunk_types": [dict(ct) for ct in chunk_types],
            "academic_focus": "Frankfurt School critical theory, dialectical materialism, social philosophy",
            "features": [
                "Page-accurate citations",
                "Dual embedding system (SPECTER2 + OpenAI)",
                "CHASE-SQL quality scoring",
                "Concept-based chunking"
            ]
        }
        
        tool_context.state["corpus_overview"] = overview
        return overview
        
    except Exception as e:
        logger.error(f"Failed to get corpus overview: {e}")
        return {"error": f"Overview failed: {str(e)}"}