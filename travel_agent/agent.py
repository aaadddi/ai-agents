from langgraph import graph
from utils import get_redis_client, get_redis_saver, set_env_key, get_llm
from agent_tools import store_memory_tool, retrieve_memories_tool

from langchain_core.messages import AIMessage, SystemMessage
import logging
from langchain_core.runnables.config import RunnableConfig
from langgraph_utils import LanggraphUtils, RuntimeState, create_agent
from langchain_core.messages import HumanMessage

def main(graph, thread_id: str = "book_flight", user_id: str = "demo_user"):
    """Main interaction loop for the travel agent"""

    print("Welcome to the Travel Assistant! (Type 'exit' to quit)")

    config = RunnableConfig(configurable={"thread_id": thread_id, "user_id": user_id})
    state = RuntimeState(messages=[])

    while True:
        user_input = input("\nYou (type 'quit' to quit): ")

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the Travel Assistant. Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        try:
            # Process user input through the graph
            for result in graph.stream(state, config=config, stream_mode="values"):
                state = RuntimeState(**result)

            logger.debug(f"# of messages after run: {len(state['messages'])}")

            # Find the most recent AI message, so we can print the response
            ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                message = ai_messages[-1].content
            else:
                logger.error("No AI messages after run")
                message = "I'm sorry, I couldn't process your request properly."
                # Add the error message to the state
                state["messages"].append(AIMessage(content=message))

            print(f"\nAssistant: {message}")

        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            error_message = "I'm sorry, I encountered an error processing your request."
            print(f"\nAssistant: {error_message}")
            # Add the error message to the state
            state["messages"].append(AIMessage(content=error_message))


if __name__ == "__main__":

    set_env_key()
    logger = logging.getLogger(__name__)
    tools = [store_memory_tool, retrieve_memories_tool]
    llm = get_llm(tools)
    redis_client = get_redis_client()
    redis_saver = get_redis_saver(redis_client)
    travel_agent = create_agent(tools, llm, redis_saver)
    langgraph_utils = LanggraphUtils(tools, travel_agent)
    graph = langgraph_utils.get_graph(redis_saver)


    try:
        user_id = input("Enter a user ID: ") or "demo_user"
        thread_id = input("Enter a thread ID: ") or "demo_thread"
    except Exception:
        exit()
    else:
        main(graph, thread_id, user_id)





