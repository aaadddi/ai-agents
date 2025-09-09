from typing import Optional
from a2a import logger
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.server.tasks import InMemoryTaskStore # Example task store
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication

from utils import get_config

config = get_config()
NEWS_AGENT_BASE_URL = config['NEWS_AGENT_BASE_URL']

# NewsInfoAgent: Provides news information
class NewsInfoAgent:
    """A simple agent that provides a static news headline."""
    async def get_latest_news(self, query: Optional[str] = None) -> str:
        # In a real agent, this would involve dynamic logic, API calls, etc.
        # The query parameter could be used to tailor the news.
        logger.info(f"NewsInfoAgent received query: {query}")
        return "Breaking News: AI discovers a new way to make coffee!"


class NewsInfoAgentExecutor(AgentExecutor):
    """Handles A2A requests for the NewsInfoAgent."""

    def __init__(self):
        super().__init__()
        self.agent = NewsInfoAgent()
        logger.info("NewsInfoAgentExecutor initialized.")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info(f"NewsInfoAgentExecutor executing task: {context.task_id}")
        if context.request_message:
            logger.info(f"Request message content: {context.request_message.model_dump_json(indent=2)}")

        query_text = None
        if context.request_message and context.request_message.message and context.request_message.message.parts:
            for part in context.request_message.message.parts:
                if part.kind == 'text' and hasattr(part, 'text'):
                    query_text = part.text
                    logger.info(f"Extracted query from message: {query_text}")
                    break

        try:
            news_result = await self.agent.get_latest_news(query_text)
            event_queue.enqueue_event(new_agent_text_message(news_result))
            logger.info(f"NewsInfoAgentExecutor successfully sent news: {news_result}")
        except Exception as e:
            error_message = f"Error in NewsInfoAgentExecutor: {str(e)}"
            logger.error(error_message, exc_info=True)
            event_queue.enqueue_event(new_agent_text_message(f"Sorry, an error occurred: {error_message}"))
        finally:
            event_queue.enqueue_event(None)  # Signal completion

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        logger.warning(f"NewsInfoAgentExecutor received cancel request for task: {context.task_id}, but cancel is not supported.")
        event_queue.enqueue_event(new_agent_text_message("Cancel operation is not supported by this agent."))
        event_queue.enqueue_event(None)  # Signal completion

news_skill = AgentSkill(
    id='get_latest_news',
    name='Get Latest News',
    description='Provides the latest news headline.',
    tags=['news', 'information', 'tldr'],
    examples=['what is the news?', 'latest headline', 'give me news']
)

# Define the AgentCard for NewsInfoAgent
news_agent_card = AgentCard(
    name='News Information Agent',
    description='Provides news headlines for the TLDR of the day.',
    url=NEWS_AGENT_BASE_URL,  # Defined in the import cell
    version='1.0.0',
    defaultInputModes=['text'],
    defaultOutputModes=['text'],
    capabilities=AgentCapabilities(streaming=True), # Our executor supports streaming via EventQueue
    skills=[news_skill],
    supportsAuthenticatedExtendedCard=False # For simplicity, no extended card for this agent
)

logger.info(f"News Agent Card defined: {news_agent_card.name}")

# Instantiate the executor, request handler, and task store
news_agent_executor = NewsInfoAgentExecutor()
news_task_store = InMemoryTaskStore()
news_request_handler = DefaultRequestHandler(
    agent_executor=news_agent_executor,
    task_store=news_task_store,
)

# Create the A2A Starlette Application for the News Agent
news_agent_server_app = A2AStarletteApplication(
    agent_card=news_agent_card,
    http_handler=news_request_handler,
    # extended_agent_card can be provided if supportsAuthenticatedExtendedCard is True
).build() # .build() returns the Starlette app instance

logger.info(f"A2AStarletteApplication for News Agent created and built.")

print("News Agent server configuration is ready. ",
      "See comments above on how to create a separate Python script to run it using uvicorn. ",
      f"It should listen on port 9001 as per NEWS_AGENT_BASE_URL ({NEWS_AGENT_BASE_URL}).")
