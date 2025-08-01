"""
Academic Database Agent for Eric's Corpus Integration

This module provides PostgreSQL database access to Eric's academic corpus
for the Google ADK data science agent, replacing BigQuery functionality
with CHASE-SQL powered academic research capabilities.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import asyncio
import sys

# Add Eric's corpus processor to path
sys.path.append('/home/scott/research/eric-corpus-processor/src')

try:
    from eric_corpus.database.postgresql_adapter import PostgreSQLAdapter
except ImportError as e:
    logging.error(f"Failed to import PostgreSQLAdapter: {e}")
    PostgreSQLAdapter = None

logger = logging.getLogger(__name__)


class AcademicDatabaseAgent:
    """Academic database agent for Eric's corpus with CHASE-SQL integration."""
    
    def __init__(self):
        """Initialize the academic database agent."""
        self.adapter = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize PostgreSQL connection to Eric's corpus."""
        if PostgreSQLAdapter is None:
            raise ImportError("PostgreSQLAdapter not available")
            
        # PostgreSQL configuration from environment/project settings
        config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'paper_chunks'),
            'user': os.getenv('POSTGRES_USER', 'eric_production'),
            'password': os.getenv('POSTGRES_PASSWORD', 'secure_production_2025')
        }
        
        try:
            self.adapter = PostgreSQLAdapter(config)
            logger.info("Academic database connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize academic database: {e}")
            raise
    
    def get_schema_description(self) -> str:
        """Get a description of Eric's academic database schema for the ADK agent."""
        return """
        Eric's Academic Corpus Schema:
        
        1. papers - Main documents table
           - paper_id: Unique document identifier
           - title: Document title
           - authors: Array of author names
           - publication_year: Publication year
           - doi: Digital Object Identifier  
           - abstract: Document abstract
        
        2. paper_chunks - Page-accurate document chunks
           - chunk_id: Unique chunk identifier
           - paper_id: Reference to parent document
           - paper_title: Document title for reference
           - title: Chunk title/section heading
           - content: Chunk text content
           - section_number: Section numbering
           - section_title: Section heading
           - chunk_type: Type of content (introduction, methodology, conclusion, etc.)
           - page_numbers: Array of page numbers this chunk spans
           - word_count: Number of words in chunk
           - academic_quality_score: CHASE-SQL quality score (0.0-1.0)
           - specter2_embedding: Academic similarity embedding (768D)  
           - openai_embedding: General semantic embedding (3072D)
           - created_at: Processing timestamp
        
        3. chunk_metadata - Additional chunk metadata
           - chunk_id: Reference to chunk
           - keywords: Extracted keywords array
           - key_concepts: Key conceptual terms
           - citations: Referenced citations in chunk
        
        Key Features:
        - Page-accurate chunking with precise page number tracking
        - Dual embedding system for academic and general similarity search
        - CHASE-SQL quality scoring for academic compliance
        - Citation tracking with academic formatting
        - Full-text search capabilities across all academic content
        """
    
    async def execute_academic_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an academic research query using CHASE-SQL methodology.
        
        Args:
            query: Natural language research query
            context: Additional context for query execution
            
        Returns:
            Dictionary containing query results and metadata
        """
        try:
            # For now, implement basic SQL execution
            # TODO: Integrate full CHASE-SQL multi-path generation
            
            conn = self.adapter.get_connection()
            results = {}
            
            with conn.cursor() as cursor:
                # Example academic queries based on common research patterns
                if any(term in query.lower() for term in ['papers', 'documents', 'corpus']):
                    cursor.execute("""
                        SELECT paper_id, title, authors, publication_year, abstract
                        FROM papers 
                        ORDER BY publication_year DESC
                        LIMIT 20
                    """)
                    papers = cursor.fetchall()
                    results['papers'] = [dict(zip([desc[0] for desc in cursor.description], paper)) for paper in papers]
                    results['type'] = 'paper_list'
                    
                elif any(term in query.lower() for term in ['citations', 'references']):
                    cursor.execute("""
                        SELECT c.chunk_id, c.paper_title, c.title, c.content, 
                               cm.citations, c.page_numbers
                        FROM paper_chunks c
                        LEFT JOIN chunk_metadata cm ON c.chunk_id = cm.chunk_id
                        WHERE cm.citations IS NOT NULL AND cm.citations != '[]'
                        LIMIT 10
                    """)
                    citations = cursor.fetchall()
                    results['citations'] = [dict(zip([desc[0] for desc in cursor.description], citation)) for citation in citations]
                    results['type'] = 'citation_list'
                    
                elif any(term in query.lower() for term in ['concepts', 'keywords', 'themes']):
                    cursor.execute("""
                        SELECT c.chunk_id, c.paper_title, c.title, c.content,
                               cm.key_concepts, cm.keywords, c.page_numbers
                        FROM paper_chunks c
                        LEFT JOIN chunk_metadata cm ON c.chunk_id = cm.chunk_id
                        WHERE cm.key_concepts IS NOT NULL AND cm.key_concepts != '[]'
                        LIMIT 10
                    """)
                    concepts = cursor.fetchall()
                    results['concepts'] = [dict(zip([desc[0] for desc in cursor.description], concept)) for concept in concepts]
                    results['type'] = 'concept_list'
                    
                else:
                    # Default: return corpus overview
                    cursor.execute("""
                        SELECT 
                            COUNT(DISTINCT p.paper_id) as total_papers,
                            COUNT(DISTINCT c.chunk_id) as total_chunks,
                            COUNT(DISTINCT CASE WHEN c.specter2_embedding IS NOT NULL THEN c.chunk_id END) as specter2_embeddings,
                            COUNT(DISTINCT CASE WHEN c.openai_embedding IS NOT NULL THEN c.chunk_id END) as openai_embeddings,
                            MIN(p.publication_year) as earliest_year,
                            MAX(p.publication_year) as latest_year
                        FROM papers p
                        LEFT JOIN paper_chunks c ON p.paper_id = c.paper_id
                    """)
                    stats = cursor.fetchone()
                    results['corpus_stats'] = dict(zip([desc[0] for desc in cursor.description], stats))
                    results['type'] = 'corpus_overview'
            
            self.adapter.return_connection(conn)
            
            # Add metadata
            results['query'] = query
            results['timestamp'] = datetime.now().isoformat()
            results['source'] = 'eric_academic_corpus'
            
            return results
            
        except Exception as e:
            logger.error(f"Academic query execution failed: {e}")
            return {
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    async def search_similar_content(self, query_text: str, embedding_type: str = 'openai', 
                                   limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar academic content using vector embeddings.
        
        Args:
            query_text: Text to search for
            embedding_type: 'specter2' for academic similarity, 'openai' for general
            limit: Maximum number of results
            
        Returns:
            List of similar chunks with similarity scores
        """
        try:
            # For now, return a placeholder - will implement full vector search
            # TODO: Generate embedding for query_text and use adapter.search_similar_chunks
            
            conn = self.adapter.get_connection()
            with conn.cursor() as cursor:
                # Temporary: Use full-text search as fallback
                cursor.execute("""
                    SELECT chunk_id, paper_title, title, content, page_numbers,
                           academic_quality_score
                    FROM paper_chunks
                    WHERE content ILIKE %s
                    ORDER BY academic_quality_score DESC
                    LIMIT %s
                """, (f'%{query_text}%', limit))
                
                results = cursor.fetchall()
                similar_chunks = []
                
                for result in results:
                    similar_chunks.append({
                        'chunk_id': result[0],
                        'paper_title': result[1],
                        'title': result[2],
                        'content': result[3][:500] + '...' if len(result[3]) > 500 else result[3],
                        'page_numbers': result[4],
                        'quality_score': result[5],
                        'similarity_type': embedding_type
                    })
            
            self.adapter.return_connection(conn)
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific paper."""
        try:
            conn = self.adapter.get_connection()
            
            with conn.cursor() as cursor:
                # Get paper metadata
                cursor.execute("""
                    SELECT paper_id, title, authors, publication_year, doi, abstract
                    FROM papers
                    WHERE paper_id = %s
                """, (paper_id,))
                
                paper = cursor.fetchone()
                if not paper:
                    return {'error': 'Paper not found'}
                
                paper_data = dict(zip([desc[0] for desc in cursor.description], paper))
                
                # Get chunks for this paper
                cursor.execute("""
                    SELECT chunk_id, title, content, section_number, section_title,
                           chunk_type, page_numbers, word_count, academic_quality_score
                    FROM paper_chunks
                    WHERE paper_id = %s
                    ORDER BY section_number, created_at
                """, (paper_id,))
                
                chunks = cursor.fetchall()
                paper_data['chunks'] = [dict(zip([desc[0] for desc in cursor.description], chunk)) for chunk in chunks]
                paper_data['chunk_count'] = len(chunks)
            
            self.adapter.return_connection(conn)
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to get paper details: {e}")
            return {'error': str(e)}
    
    async def get_citation_analysis(self, search_term: str = None) -> Dict[str, Any]:
        """Get citation analysis across the corpus."""
        try:
            conn = self.adapter.get_connection()
            
            with conn.cursor() as cursor:
                if search_term:
                    cursor.execute("""
                        SELECT c.paper_title, c.title, c.content, cm.citations, c.page_numbers
                        FROM paper_chunks c
                        LEFT JOIN chunk_metadata cm ON c.chunk_id = cm.chunk_id
                        WHERE (c.content ILIKE %s OR c.title ILIKE %s)
                          AND cm.citations IS NOT NULL 
                          AND cm.citations != '[]'
                        ORDER BY c.academic_quality_score DESC
                        LIMIT 20
                    """, (f'%{search_term}%', f'%{search_term}%'))
                else:
                    cursor.execute("""
                        SELECT c.paper_title, c.title, c.content, cm.citations, c.page_numbers
                        FROM paper_chunks c
                        LEFT JOIN chunk_metadata cm ON c.chunk_id = cm.chunk_id
                        WHERE cm.citations IS NOT NULL AND cm.citations != '[]'
                        ORDER BY c.academic_quality_score DESC
                        LIMIT 20
                    """)
                
                citations = cursor.fetchall()
                citation_data = [dict(zip([desc[0] for desc in cursor.description], citation)) for citation in citations]
            
            self.adapter.return_connection(conn)
            
            return {
                'citation_analysis': citation_data,
                'search_term': search_term,
                'total_found': len(citation_data)
            }
            
        except Exception as e:
            logger.error(f"Citation analysis failed: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close database connection."""
        if self.adapter:
            self.adapter.close()


# Global instance for ADK integration
academic_db_agent = None

def get_academic_agent() -> AcademicDatabaseAgent:
    """Get or create the global academic database agent instance."""
    global academic_db_agent
    if academic_db_agent is None:
        academic_db_agent = AcademicDatabaseAgent()
    return academic_db_agent