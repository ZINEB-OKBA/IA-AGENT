from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage
import asyncio

load_dotenv(override=True)

mcp_client = MultiServerMCPClient(
    {
        "mcp_server": {
            "transport": "streamable_http",
            "url": "http://localhost:23000/mcp",
        }
    }
)

async def main():
    tools = await mcp_client.get_tools()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="answer the user question using provided tools",
    )

    while True:
        user_query = input("Question: ")

        if user_query == "exit":
            break

        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}
        )

        print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())