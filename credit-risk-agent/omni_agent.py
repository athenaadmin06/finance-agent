import asyncio
import os
import json
from groq import Groq
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

from petstore_api import petstore_tools_schema, petstore_functions

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def run_omni_agent(user_prompt):
    """
    Connects to the local MCP server via SSE for Hacker News tools, 
    loads local Petstore API tools, and uses the Groq LLM to answer the prompt.
    """
    try:
        # 1. Connect to FastMCP Server running locally for Hacker News tools
        async with sse_client("http://localhost:8000/sse") as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Format MCP Tools for Groq
                mcp_tools_raw = await session.list_tools()
                mcp_tools = [{
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                } for t in mcp_tools_raw.tools]

                # 2. Combine MCP Tools with Local Petstore Tools
                all_tools = mcp_tools + petstore_tools_schema

                # 3. Setup Conversation Loop
                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are a helpful Omni-Agent assistant. YOU CAN DO TWO THINGS: "
                            "1. Lookup Hacker News top stories using MCP tools. "
                            "2. Lookup pets in a pet store using API Gateway tools. "
                            "Always answer queries accurately by using combinations of these tools."
                        )
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ]

                # 4. Agent Tool-Calling Loop
                while True:
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile", 
                        messages=messages,
                        tools=all_tools,
                        tool_choice="auto" 
                    )

                    response_message = response.choices[0].message

                    if not response_message.tool_calls:
                        return response_message.content

                    messages.append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in response_message.tool_calls
                        ]
                    })
                    
                    # Intercept and redirect tool calls dynamically
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        tool_output = ""
                        if tool_name in petstore_functions:
                            # Route to python pet store functions
                            func = petstore_functions[tool_name]
                            result = func(**tool_args)
                            tool_output = json.dumps(result)
                        else:
                            # Route to the MCP SSE Server
                            result = await session.call_tool(tool_name, tool_args)
                            if result.content:
                                tool_output = result.content[0].text
                            else:
                                tool_output = "No results returned."

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(tool_output)
                        })

    except Exception as e:
        return f"🚨 Error running Omni-Agent: {str(e)}\n\n(Tip: Did you ensure `mcp_server.py` is running in another terminal?)"
