import asyncio
import json
import os
from openai import AsyncOpenAI
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool



async def main( ):
    server_parameters = StdioServerParameters(command="python",args=["server.py"])
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools_response = await session.list_tools()
            langchain_tool = []
            for i in mcp_tool_response.tool:
                tool = Tool(
                    name = i.name,
                    description = i.description,
                    func = lambda **kwargs: asyncio.run(session.call_tool(i.name, arguments=kwargs))
                )
                langchain_tool.append(tool)

            llm = ChatOpenAI(model="gpt-4o", api_key="YOUR_KEY")
            llm_with_tools = llm.bind_tools(langchain_tool)
            while True:
                text = input("your prompt here")
                response = await llm_with_tools.invoke(text)
                print(response.tool_calls)
                print(response.content)

if __name__ =="__main__":
     asyncio.run(main())