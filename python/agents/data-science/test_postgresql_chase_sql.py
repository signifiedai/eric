"""
Test for PostgreSQL CHASE-SQL integration issue
Following London TDD approach: Outside-In, Behavior-Driven Testing
"""

import pytest
from unittest.mock import Mock, patch
from data_science.sub_agents.postgresql.tools import academic_nl2sql
from google.adk.tools import ToolContext


def test_academic_nl2sql_should_use_postgresql_schema_not_bigquery():
    """
    FAILING TEST (London TDD Step 1)
    
    This test demonstrates the current issue:
    PostgreSQL agent is trying to access BigQuery schema fields
    """
    # Arrange - Set up PostgreSQL context
    tool_context = Mock(spec=ToolContext)
    tool_context.state = {
        "database_settings": {
            "pg_schema": "PostgreSQL Schema Description",
            "corpus_type": "eric_academic",
            # Note: No 'bq_ddl_schema' key - this should work
        }
    }
    
    question = "What papers are available?"
    
    # Act & Assert - This should NOT fail with KeyError: 'bq_ddl_schema'
    try:
        result = academic_nl2sql(question, tool_context)
        assert result is not None
        assert "SELECT" in result.upper()
        print("✅ Test PASSED - PostgreSQL schema accessed correctly")
    except KeyError as e:
        if "'bq_ddl_schema'" in str(e):
            print(f"❌ Test FAILED - Still trying to access BigQuery schema: {e}")
            pytest.fail(f"PostgreSQL agent incorrectly accessing BigQuery schema: {e}")
        else:
            raise


def test_chase_sql_should_work_with_postgresql_context():
    """
    Test that CHASE-SQL tools work with PostgreSQL database settings
    """
    # Arrange
    tool_context = Mock(spec=ToolContext)
    tool_context.state = {
        "database_settings": {
            "pg_schema": "Eric's Academic Corpus Schema",
            "corpus_type": "eric_academic",
            "total_papers": 111,
            "total_chunks": 31847
        },
        "all_db_settings": {
            "use_database": "PostgreSQL"
        }
    }
    
    # Act
    with patch('data_science.sub_agents.postgresql.tools.llm_client') as mock_client:
        mock_client.models.generate_content.return_value.text = "SELECT * FROM papers LIMIT 10;"
        
        result = academic_nl2sql("Show me some papers", tool_context)
        
        # Assert
        assert result == "SELECT * FROM papers LIMIT 10;"
        assert tool_context.state["sql_query"] == result


if __name__ == "__main__":
    # Run the failing test
    test_academic_nl2sql_should_use_postgresql_schema_not_bigquery()
    test_chase_sql_should_work_with_postgresql_context()