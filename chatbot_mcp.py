from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": r"C:\Users\umapc\Downloads\langgraph\myenv\Scripts\python.exe",
            "args": [
                r"C:\Users\umapc\Downloads\langgraph\myenv\mcp_server.py"
            ]
        }
    }
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", END: END})
    graph.add_edge("tool_node", "chat_node")
    return graph.compile()

async def main():
    chatbot = await build_graph()
    next_state = {"messages": [HumanMessage(content="what is the multiplication of 2334 and 42 and give answer like a cricket commentator")]}
    final_state2 = await chatbot.ainvoke(next_state)
    print(final_state2["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main()) 