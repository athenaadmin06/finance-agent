import requests
from mcp.server.fastmcp import FastMCP

# Initialize the server
mcp = FastMCP("hacker_news_digest")

# Base URL for the free Hacker News Firebase API
HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"

@mcp.tool()
def get_top_stories(limit: int = 10) -> list[int]:
    """
    Fetches the IDs of the current top stories on Hacker News.
    Use this first to find out what is currently trending.
    """
    try:
        response = requests.get(f"{HN_BASE_URL}/topstories.json")
        response.raise_for_status()
        return response.json()[:limit]
    except Exception as e:
        return f"Error fetching top stories: {str(e)}"

@mcp.tool()
def get_story_details(story_id: int) -> dict:
    """
    Fetches the details of a specific Hacker News story using its ID.
    Returns the title, author, score, and URL.
    """
    try:
        response = requests.get(f"{HN_BASE_URL}/item/{story_id}.json")
        response.raise_for_status()
        data = response.json()
        
        return {
            "title": data.get("title"),
            "author": data.get("by"),
            "score": data.get("score"),
            "url": data.get("url", "No URL (Ask HN post)"),
            "comments_count": data.get("descendants", 0)
        }
    except Exception as e:
        return {"error": f"Error fetching story details: {str(e)}"}

if __name__ == "__main__":
    # Run the server communicating over SSE (defaults to port 8000)
    mcp.run(transport="sse")