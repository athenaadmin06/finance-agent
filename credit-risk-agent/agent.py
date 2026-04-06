import asyncio
import os
import json
from groq import Groq
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def run_agent():
    # Make sure your server.py is running on port 8000!
    async with sse_client("http://localhost:8000/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # 1. Map tools correctly
            mcp_tools = await session.list_tools()
            groq_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]

            # 2. PROMPT
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a professional security researcher. You must use tools to gather and process information. "
                        "First, search for news. Second, optimize the links. Third, summarize for LinkedIn. "
                        "Finally, YOU MUST provide the complete, optimized LinkedIn posts in your final response to the user."
                    )
                },
                {
                    "role": "user", 
                    "content": "Find news about 'LLM prompt injection', optimize the links, and draft a LinkedIn post."
                }
            ]

            # 3. The Execution Loop
            try:
                while True:
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile", 
                        messages=messages,
                        tools=groq_tools,
                        tool_choice="auto" 
                    )

                    response_message = response.choices[0].message

                    if not response_message.tool_calls:
                        print(f"\n--- GRAFYN AI AGENT OUTPUT ---\n{response_message.content}")
                        break

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
                    
                    # --- Update this section in your agent.py ---
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        print(f"[*] Calling Tool: {tool_name} with {tool_args}")
                        
                        # Execute on MCP Server
                        result = await session.call_tool(tool_name, tool_args)
                        
                        # DEFENSIVE CHECK: Ensure result.content is not empty
                        tool_output = "No results returned by tool."
                        if result.content:
                            tool_output = result.content[0].text
                        else:
                            print(f"[!] Warning: Tool {tool_name} returned no content.")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(tool_output)
                        })

            except Exception as e:
                print(f"[!] Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_agent())