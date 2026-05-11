from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv(override=True)

# Création du serveur MCP
mcp = FastMCP(
    name="mcp-server",
    host="0.0.0.0",
    port=23000
)

# Client Tavily
web_search_client = TavilySearch()


@mcp.tool()
def get_employee_infos(name: str):
    """
    Get information about a given employee
    """
    return {
        "name": name,
        "salary": 21000,
        "seniority": 12
    }


@mcp.tool()
def web_search(query: str):
    """
    Search the web for the given query
    """
    results = web_search_client.invoke({"query": query})
    return results


if __name__ == "__main__":
    mcp.run(transport="streamable-http")