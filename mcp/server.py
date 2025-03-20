from mcp.server import FastMCP

mcp = FastMCP(name='demo', log_level='DEBUG')


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


# # Add a dynamic greeting resource
# @mymcp.resource("greeting://{name}")
# def get_greeting(name: str) -> str:
#     """Get a personalized greeting"""
#     return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="sse")
