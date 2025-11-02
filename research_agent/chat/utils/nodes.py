from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState
from ...utils import get_llm
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults()
llm = get_llm()

def initialize_node(state: MessagesState):
    user_input = state["messages"][-1].content
    print(f"\n[🟦 Initialize Node] Extracting topic from user input: {user_input}\n")

    topic_prompt = f"""
    You are an AI research assistant.

    Your task: read the following user request and generate only a short, research-suitable title (5–10 words max) that captures the core topic.

    User request: "{user_input}"

    Output ONLY the title — no explanation.
    """
    topic = llm.invoke([HumanMessage(content=topic_prompt)]).content.strip()

    response = AIMessage(content=f"Got it. Starting research on: **{topic}**")
    print(f"[🤖 AI] {response.content}\n")

    return {"messages": state["messages"] + [response]}


# -----------------------------
# Node 2: Clarify
# -----------------------------
def clarify_node(state: MessagesState):
    last_ai = state["messages"][-1].content
    print(f"[🟨 Clarify Node] Asking clarifying questions for topic context...\n")

    clarify_prompt = f"""
    You are preparing to research this topic: {last_ai}.
    Before starting, ask 2-3 clarifying questions to understand the user's focus.
    Example: “Should I focus on health, environmental, or cultural aspects?”
    """
    clarification = llm.invoke([HumanMessage(content=clarify_prompt)]).content
    print(f"[🤖 AI] {clarification}\n")

    return {"messages": state["messages"] + [AIMessage(content=clarification)]}


# -----------------------------
# Node 3: Research
# -----------------------------
def research_node(state: MessagesState):
    print(f"[🟩 Research Node] Searching and summarizing information...\n")

    topic = None
    for msg in reversed(state["messages"]):
        if "Starting research on" in msg.content:
            topic = msg.content.split("**")[1]
            break

    if not topic:
        print("[⚠️] No topic found in messages.")
        return state

    search_results = search_tool.run(topic)
    search_text = "\n".join(search_results) if isinstance(search_results, list) else search_results

    summary_prompt = f"""
    Summarize key insights about '{topic}' from these search results.
    Focus on facts, studies, or evidence.
    Results:\n{search_text[:4000]}
    """
    summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
    print(f"[🤖 AI] Summary generated for topic '{topic}'.\n")

    return {"messages": state["messages"] + [AIMessage(content=summary)]}


# -----------------------------
# Node 4: Final Report
# -----------------------------
def report_node(state: MessagesState):
    print(f"[🟪 Report Node] Preparing final summary...\n")
    report = f"🧾 Final Research Summary:\n\n{state['messages'][-1].content}"
    print(f"[🤖 AI] {report}\n")
    return {"messages": state["messages"] + [AIMessage(content=report)]}
