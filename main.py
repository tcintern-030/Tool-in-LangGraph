import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from tools import tools

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[
        list,
        add_messages
    ]

def agent_node(state: State):
    print("---AGENT NODE---")

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    print("Agent Response:")

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                "Tool Selected:",
                tool_call["name"]
            )
            print(
                "Arguments:",
                tool_call["args"]
            )
    else:
        print(response.content)

    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        print("\nRouting to Tool Node...")
        return "tools"

    print("\nNo tool required.")

    return END

graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START,"agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

graph_builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def run_agent(question):
    print("---USER---")
    print(question)

    config = {
        "configurable": {
            "thread_id": "conversation_1"
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ]
        }, 
        config
    )

    final_message = result["messages"][-1]

    print("---FINAL ANSWER---")

    if isinstance(final_message.content, list):
        for item in final_message.content:
            if item.get("type") == "text":
                print(item["text"])

    else:
        print(final_message.content)


if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk something "
            "(type exit to stop): "
        )

        if question.lower() == "exit":
            break

        run_agent(question)