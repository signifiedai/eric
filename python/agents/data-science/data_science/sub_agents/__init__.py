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

import os

from .bqml.agent import root_agent as bqml_agent
from .analytics.agent import root_agent as ds_agent
from .bigquery.agent import database_agent as bigquery_agent

# Import PostgreSQL agent
try:
    from .postgresql.agent import academic_database_agent as postgresql_agent
    POSTGRESQL_AVAILABLE = True
except ImportError:
    postgresql_agent = None
    POSTGRESQL_AVAILABLE = False

# Dynamic DB agent selection based on environment
def get_db_agent():
    """Get the appropriate database agent based on configuration."""
    use_eric_corpus = os.getenv("USE_ERIC_CORPUS", "false").lower() == "true"
    
    if use_eric_corpus and POSTGRESQL_AVAILABLE:
        return postgresql_agent
    else:
        return bigquery_agent

# Default to BigQuery agent for backward compatibility
db_agent = get_db_agent()

__all__ = ["bqml_agent", "ds_agent", "db_agent", "bigquery_agent"]
if POSTGRESQL_AVAILABLE:
    __all__.append("postgresql_agent")
