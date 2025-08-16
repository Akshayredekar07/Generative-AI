import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


async def main():
    # Get keys from environment variables
    gemini_key = os.getenv("GOOGLE_API_KEY")
    owm_key = os.getenv("OPENWEATHER_API_KEY")

    if not gemini_key:
        raise ValueError("GOOGLE_API_KEY not found in .env or environment variables.")
    if not owm_key:
        raise ValueError("OPENWEATHER_API_KEY not found in .env or environment variables.")

    # Configure MCP servers
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "stdio",
                "command": r"D:\Langchain\Generative-AI\langraph-openwather-mcp\mcp-openweather\mcp-weather.exe",
                "args": [],
                "env": {"OPENWEATHER_API_KEY": owm_key}
            },
            "calculator": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_server_calculator"]
            }
        }
    )

    # Load tools from MCP servers
    tools = await client.get_tools()

    # Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    # Build LangGraph workflow
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()

    print("\n--- Weather & Calculator Agent ---")

    while True:
        user_question = input("\nAsk me anything (weather or calculation) → ")
        if user_question.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("\n--- Agent is thinking... ---")
        result = await graph.ainvoke({"messages": user_question})
        print("\n--- Answer ---")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
