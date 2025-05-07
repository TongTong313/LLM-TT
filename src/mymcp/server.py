from mcp.server import FastMCP
import mcp
import asyncio
from typing import Callable, Optional, List, Any
from mymcp.tool import get_current_time, add, baidu_search, get_current_weather
from mymcp.mcp_adapter import MCPOpenAIAdapter


class MCPServer:
    """MCP server"""

    def __init__(self, name: str = "demo", log_level: str = "DEBUG"):
        self.server = FastMCP(name=name, log_level=log_level)

    def register_tool(self,
                      tool: Callable,
                      name: Optional[str] = None,
                      description: Optional[str] = None):
        """注册工具
        
        Args:
            tool (Callable): 工具函数
            name (str, optional): 工具名称，可选
            description (str, optional): 工具描述，可选
        """
        self.server.add_tool(tool, name=name, description=description)

    async def run(self, transport: Optional[str] = "stdio"):
        """运行MCP服务器
        
        Args:
            transport (str, optional): 传输方式，可选值为 "stdio" 或 "sse"，默认值为 "stdio"。
        """
        if transport == "stdio":
            await self.server.run_stdio_async()
        else:
            raise ValueError(f"Unsupported transport: {transport}")


async def main():
    mymcp = MCPServer()
    mymcp.register_tool(get_current_time, name="get_current_time")
    mymcp.register_tool(add, name="add")
    mymcp.register_tool(baidu_search, name="baidu_search")
    mymcp.register_tool(get_current_weather, name="get_current_weather")

    mcp_adapter = MCPOpenAIAdapter()
    tools = await mymcp.server.list_tools()
    print("MCP工具列表：")
    print(tools)

    tool_schemas = mcp_adapter.convert_to_tool_schema(tools)
    print("\n工具schema：")
    print(tool_schemas)

    # 运行MCP服务器
    await mymcp.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
