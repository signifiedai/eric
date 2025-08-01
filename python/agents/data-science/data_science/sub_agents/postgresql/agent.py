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

"""Academic Database Agent: Access Eric's corpus using PostgreSQL with CHASE-SQL."""

import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from . import tools
from .prompts import return_instructions_postgresql

NL2SQL_METHOD = os.getenv("NL2SQL_METHOD", "CHASE")  # Default to CHASE for academic queries


def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup the academic database agent."""
    
    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = tools.get_postgresql_database_settings()


academic_database_agent = Agent(
    model=os.getenv("BIGQUERY_AGENT_MODEL", "gemini-2.0-flash-exp"),  # Use same model as BigQuery agent
    name="academic_database_agent",
    instruction=return_instructions_postgresql(),
    tools=[
        tools.academic_nl2sql,
        tools.run_postgresql_validation,
        tools.search_academic_content,
        tools.get_corpus_overview,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)