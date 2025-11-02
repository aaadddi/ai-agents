from langchain.schema import HumanMessage
from .utils.nodes import initialize_node, clarify_node, research_node, report_node
from langgraph.graph import StateGraph, MessagesState, START, END

state = MessagesState()

graph = StateGraph(MessagesState)
graph.add_node("initialize", initialize_node)
graph.add_node("clarify", clarify_node)
graph.add_node("research", research_node)
graph.add_node("report", report_node)

graph.add_edge(START, "initialize")
graph.add_edge("initialize", "clarify")
graph.add_edge("clarify", "research")
graph.add_edge("research", "report")
graph.add_edge("report", END)

app = graph.compile()

def chat():
    state = {"messages": [HumanMessage(content=input("Enter your research topic: "))]}
    while True:
        state = app.invoke(state)
        print(f"\nAI: {state['messages'][-1].content}\n")

        if "Final Research Summary" in state["messages"][-1].content:
            break

        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        state["messages"].append(HumanMessage(content=user_input))

chat()


if __name__ == "__main__":
    chat()
    

