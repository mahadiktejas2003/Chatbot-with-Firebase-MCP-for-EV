from pydantic_ai import RunContext, Tool as PydanticTool
from pydantic_ai.tools import ToolDefinition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from contextlib import AsyncExitStack
from typing import Any, List, Dict, Union
import asyncio
import logging
import shutil
import json
import os

logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MCPClient:
    """Manages connections to one or more MCP servers based on mcp_config.json"""

    def __init__(self) -> None:
        self.servers: List[MCPServer] = []
        self.config: dict[str, Any] = {}
        self.tools: List[Any] = []
        self.exit_stack = AsyncExitStack()

    def load_servers(self, config_path: str) -> None:
        """Load server configuration from a JSON file (typically mcp_config.json)
        and creates an instance of each server (no active connection until 'start' though).

        Args:
            config_path: Path to the JSON configuration file.
        """
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.servers = [MCPServer(name, config) for name, config in self.config["mcpServers"].items()]

    async def start(self) -> List[PydanticTool]:
        """Starts each MCP server and returns the tools for each server formatted for Pydantic AI."""
        self.tools = []
        for server in self.servers:
            try:
                logging.debug(f"Initializing server: {server.name}")
                await server.initialize()
                tools = await server.create_pydantic_ai_tools()
                self.tools += tools
                logging.debug(f"Server {server.name} initialized successfully with {len(tools)} tools")
            except Exception as e:
                logging.error(f"Failed to initialize server {server.name}: {e}")
                await self.cleanup_servers()
                return []

        return self.tools

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during cleanup of server {server.name}: {e}")

    async def cleanup(self) -> None:
        """Clean up all resources including the exit stack."""
        try:
            # First clean up all servers
            await self.cleanup_servers()
            # Then close the exit stack
            await self.exit_stack.aclose()
        except Exception as e:
            logging.warning(f"Warning during final cleanup: {e}")


class MCPServer:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=self.config["env"]
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def create_pydantic_ai_tools(self) -> List[PydanticTool]:
        """Convert MCP tools to pydantic_ai Tools."""
        tools = (await self.session.list_tools()).tools
        return [self.create_tool_instance(tool) for tool in tools]            

    def create_tool_instance(self, tool: MCPTool) -> PydanticTool:
        """Initialize a Pydantic AI Tool from an MCP Tool."""
        async def execute_tool(**kwargs: Any) -> str:
            """
            Execute the MCP tool and handle different response types.
            Always returns a string to ensure compatibility with the calling code.
            """
            try:
                result = await self.session.call_tool(tool.name, arguments=kwargs)
                
                # Handle different result formats and convert to string
                if isinstance(result, dict):
                    # Try to pretty-print JSON if possible
                    try:
                        return json.dumps(result, indent=2)
                    except:
                        return str(result)
                elif hasattr(result, 'to_dict'):
                    # If it has a to_dict method, use it
                    return str(result.to_dict())
                elif hasattr(result, '__dict__'):
                    # For objects with a __dict__, use that
                    return str(result.__dict__)
                else:
                    # Default case, just convert to string
                    return str(result)
            except Exception as e:
                logging.error(f"Error executing tool {tool.name}: {e}")
                return f"Error executing {tool.name}: {str(e)}"

        async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            """
            Ensure proper setup of tool definition with input schema.
            This prevents issues with schema validation.
            """
            try:
                tool_def.parameters_json_schema = tool.inputSchema
                return tool_def
            except Exception as e:
                logging.error(f"Error preparing tool {tool.name}: {e}")
                return None
        
        return PydanticTool(
            execute_tool,
            name=tool.name,
            description=tool.description or f"Tool for {tool.name}",
            takes_ctx=False,
            prepare=prepare_tool
        )

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")