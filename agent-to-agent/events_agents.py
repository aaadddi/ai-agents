from typing import Optional
from agent import logger
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
EVENTS_AGENT_BASE_URL = config['EVENTS_AGENT_BASE_URL']


# EventsInfoAgent: Provides event information
class EventsInfoAgent:
    """A simple agent that provides a static event update."""
    async def get_current_events(self, query: Optional[str] = None) -> str:
        # In a real agent, this would involve dynamic logic, API calls, etc.
        logger.info(f"EventsInfoAgent received query: {query}")
        return "Current Event: The annual 'Innovate AI' conference is happening this week!"

# EventsInfoAgentExecutor: Implements the A2A AgentExecutor interface
class EventsInfoAgentExecutor(AgentExecutor):
    """Handles A2A requests for the EventsInfoAgent."""

    def __init__(self):
        super().__init__()
        self.agent = EventsInfoAgent()
        logger.info("EventsInfoAgentExecutor initialized.")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info(f"EventsInfoAgentExecutor executing task: {context.task_id}")
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
            event_result = await self.agent.get_current_events(query_text)
            event_queue.enqueue_event(new_agent_text_message(event_result))
            logger.info(f"EventsInfoAgentExecutor successfully sent event info: {event_result}")
        except Exception as e:
            error_message = f"Error in EventsInfoAgentExecutor: {str(e)}"
            logger.error(error_message, exc_info=True)
            event_queue.enqueue_event(new_agent_text_message(f"Sorry, an error occurred: {error_message}"))
        finally:
            event_queue.enqueue_event(None) # Signal completion

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        logger.warning(f"EventsInfoAgentExecutor received cancel request for task: {context.task_id}, but cancel is not supported.")
        event_queue.enqueue_event(new_agent_text_message("Cancel operation is not supported by this agent."))
        event_queue.enqueue_event(None) # Signal completion

logger.info("EventsInfoAgent and EventsInfoAgentExecutor classes defined.")


events_skill = AgentSkill(
    id='get_current_events',
    name='Get Current Events',
    description='Provides information about current events.',
    tags=['events', 'information', 'tldr', 'conference'],
    examples=['what are the current events?', 'any ongoing events?', 'tell me about events']
)

# Define the AgentCard for EventsInfoAgent
events_agent_card = AgentCard(
    name='Current Events Information Agent',
    description='Provides updates on current events for the TLDR of the day.',
    url=EVENTS_AGENT_BASE_URL,  # Defined in the import cell
    version='1.0.0',
    defaultInputModes=['text'],
    defaultOutputModes=['text'],
    capabilities=AgentCapabilities(streaming=True), # Supports streaming
    skills=[events_skill],
    supportsAuthenticatedExtendedCard=False
)

logger.info(f"Events Agent Card defined: {events_agent_card.name}")

# Instantiate the executor, request handler, and task store
events_agent_executor = EventsInfoAgentExecutor()
events_task_store = InMemoryTaskStore()
events_request_handler = DefaultRequestHandler(
    agent_executor=events_agent_executor,
    task_store=events_task_store,
)

# Create the A2A Starlette Application for the Events Agent
events_agent_server_app = A2AStarletteApplication(
    agent_card=events_agent_card,
    http_handler=events_request_handler,
).build()

logger.info(f"A2AStarletteApplication for Events Agent created and built.")
