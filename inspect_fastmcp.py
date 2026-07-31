import inspect
from mcp.server.fastmcp import FastMCP
print('FastMCP class:', FastMCP)
print('FastMCP signature:', inspect.signature(FastMCP))
print('FastMCP.run signature:', inspect.signature(FastMCP.run))
print('FastMCP.run exists:', hasattr(FastMCP, 'run'))
