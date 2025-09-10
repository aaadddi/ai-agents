from langchain_core.messages import HumanMessage
from langgraph.graph.message import MessagesState
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import RemoveMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from utils import get_llm
import logging

logger = logging.getLogger(__name__)

class RuntimeState(MessagesState):
    """Runtime state for the travel agent."""
    pass

class GraphNodes:
    
    MESSAGE_SUMMARIZATION_THRESHOLD = 6

    def __init__(self, tools, travel_agent) -> None:
        self.summarizer = get_llm()
        self.tools = tools
        self.travel_agent = travel_agent
    def respond_to_user(self,state: RuntimeState, config: RunnableConfig) -> RuntimeState:
        """Invoke the travel agent to generate a response."""
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_messages:
            logger.warning("No HumanMessage found in state")
            return state

        try:
            # Single agent invocation, not streamed (simplified for reliability)
            result = self.travel_agent.invoke({"messages": state["messages"]}, config=config)
            agent_message = result["messages"][-1]
            state["messages"].append(agent_message)
        except Exception as e:
            logger.error(f"Error invoking travel agent: {e}")
            agent_message = AIMessage(
                content="I'm sorry, I encountered an error processing your request."
            )
            state["messages"].append(agent_message)

        return state

    def execute_tools(self, state: RuntimeState, config: RunnableConfig) -> RuntimeState:
        """Execute tools specified in the latest AIMessage and append ToolMessages."""
        messages = state["messages"]
        latest_ai_message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
            None
        )

        if not latest_ai_message:
            return state  # No tool calls to process

        tool_messages = []
        for tool_call in latest_ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Find the corresponding tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                continue  # Skip if tool not found

            try:
                # Execute the tool with the provided arguments
                result = tool.invoke(tool_args, config=config)
                # Create a ToolMessage with the result
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
            except Exception as e:
                # Handle tool execution errors
                error_message = ToolMessage(
                    content=f"Error executing tool '{tool_name}': {str(e)}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_message)

        # Append the ToolMessages to the message history
        messages.extend(tool_messages)
        state["messages"] = messages
        return state


    def summarize_conversation(self,
        state: RuntimeState, config: RunnableConfig
    ) -> RuntimeState:
        """
        Summarize a list of messages into a concise summary to reduce context length
        while preserving important information.
        """
        messages = state["messages"]
        current_message_count = len(messages)
        if current_message_count < GraphNodes.MESSAGE_SUMMARIZATION_THRESHOLD:
            logger.debug(f"Not summarizing conversation: {current_message_count}")
            return state

        system_prompt = """
        You are a conversation summarizer. Create a concise summary of the previous
        conversation between a user and a travel assistant.

        The summary should:
        1. Highlight key topics, preferences, and decisions
        2. Include any specific trip details (destinations, dates, preferences)
        3. Note any outstanding questions or topics that need follow-up
        4. Be concise but informative

        Format your summary as a brief narrative paragraph.
        """

        message_content = "\n".join(
            [
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages
            ]
        )

        # Invoke the summarizer
        summary_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Please summarize this conversation:\n\n{message_content}"
            ),
        ]

        summary_response = self.summarizer.invoke(summary_messages)

        logger.info(f"Summarized {len(messages)} messages into a conversation summary")

        summary_message = SystemMessage(
            content=f"""
            Summary of the conversation so far:

            {summary_response.content}

            Please continue the conversation based on this summary and the recent messages.
            """
        )
        remove_messages = [
            RemoveMessage(id=msg.id) for msg in messages if msg.id is not None
        ]

        state["messages"] = [  # type: ignore
            *remove_messages,
            summary_message,
            state["messages"][-1],
        ]

        return state.copy()

class LanggraphUtils:
    def __init__(self,tools, travel_agent) -> None:
        self.graph_nodes = GraphNodes(tools, travel_agent)
        self.tools = tools

    def get_graph(self, redis_saver):
        workflow = StateGraph(RuntimeState)
        workflow.add_node("agent", self.graph_nodes.respond_to_user)
        workflow.add_node("execute_tools", self.graph_nodes.execute_tools)
        workflow.add_node("summarize_conversation", self.graph_nodes.summarize_conversation)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.decide_next_step,
            {"execute_tools": "execute_tools", "summarize_conversation": "summarize_conversation"},
        )
        workflow.add_edge("execute_tools", "agent")
        workflow.add_edge("summarize_conversation", END)
        graph = workflow.compile(checkpointer=redis_saver)

        return graph
    def decide_next_step(self, state):
        latest_ai_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if latest_ai_message and latest_ai_message.tool_calls:
            return "execute_tools"
        return "summarize_conversation"
    
def create_agent(tools, llm, redis_saver):
    travel_agent = create_react_agent(
        model=llm,
        tools=tools,               
        checkpointer=redis_saver, 
        prompt=SystemMessage(
            content="""
            You are a travel assistant helping users plan their trips. You remember user preferences
            and provide personalized recommendations based on past interactions.

            You have access to the following types of memory:
            1. Short-term memory: The current conversation thread
            2. Long-term memory:
            - Episodic: User preferences and past trip experiences (e.g., "User prefers window seats")
            - Semantic: General knowledge about travel destinations and requirements

            Your procedural knowledge (how to search, book flights, etc.) is built into your tools and prompts.

            Always be helpful, personal, and context-aware in your responses.
            """
        ),
        )
    return travel_agent