# Using JSON-RPC

# Agents - 
    # UserFacingAgent: This agent is responsible for interacting with the user. It will orchestrate the task of gathering TLDR information.
    # NewsInfoAgent: This agent specializes in providing the news-related part of the TLDR.
    # EventsInfoAgent: This agent specializes in providing information about current events for the TLDR.

# Standard library imports
import json
import uuid
import asyncio # For running async client code
import logging
import os

import httpx
# A2A SDK imports
from a2a.client import A2ACardResolver, A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue



from a2a.utils import new_agent_text_message # For agent executor's response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A2A_Tutorial_Notebook") # Create a logger specific to this tutorial
logger.info("A2A SDK and libraries imported. Logging configured.")

# Define base URLs for our agents (these will be separate server processes)
NEWS_AGENT_BASE_URL = "http://localhost:9001"
EVENTS_AGENT_BASE_URL = "http://localhost:9002"

# Standard paths for agent cards (as per A2A specification)
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
EXTENDED_AGENT_CARD_PATH = "/agent/authenticatedExtendedCard" # If we use extended cards

logger.info(f"News Agent will be expected at: {NEWS_AGENT_BASE_URL}")
logger.info(f"Events Agent will be expected at: {EVENTS_AGENT_BASE_URL}")





