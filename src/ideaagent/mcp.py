"""MCP (Model Context Protocol) support for IdeaAgent using FastMCP."""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from fastmcp import Client as FastMCPClient
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

load_dotenv()


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str  # Command to run the server (e.g., "npx", "python")
    args: list[str]  # Command arguments
    env: dict = field(default_factory=dict)  # Environment variables
    enabled: bool = True
    timeout: int = 30
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MCPServerConfig":
        return cls(
            name=data.get("name", "unknown"),
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            enabled=data.get("enabled", True),
            timeout=data.get("timeout", 30),
            description=data.get("description", ""),
        )

    def to_server_params(self) -> StdioServerParameters:
        """Convert to StdioServerParameters for MCP client."""
        env = os.environ.copy()
        env.update(self.env)
        
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=env,
        )


class MCPClient:
    """MCP client using FastMCP for IdeaAgent."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize MCP client.
        
        Args:
            config_path: Path to MCP configuration file
        """
        self.config_path = config_path or Path.cwd() / "mcp_config.json"
        self.servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._clients: dict[str, FastMCPClient] = {}
        
        self.enabled = os.getenv("MCP_ENABLED", "true").lower() == "true"
        
        if self.enabled:
            self.load_config()

    def load_config(self) -> None:
        """Load MCP configuration from file."""
        if not self.config_path.exists():
            self._create_default_config()
            return

        try:
            config_data = json.loads(self.config_path.read_text(encoding="utf-8"))
            for server_data in config_data.get("servers", []):
                config = MCPServerConfig.from_dict(server_data)
                self.servers[config.name] = config
        except Exception as e:
            print(f"Warning: Failed to load MCP config: {e}")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default MCP configuration."""
        default_config = {
            "version": "1.0",
            "servers": [
                {
                    "name": "filesystem",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                    "env": {},
                    "enabled": True,
                    "timeout": 30,
                    "description": "File system access for reading/writing files"
                },
                {
                    "name": "git",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-git"],
                    "env": {},
                    "enabled": False,
                    "timeout": 30,
                    "description": "Git operations support"
                }
            ]
        }
        
        try:
            self.config_path.write_text(
                json.dumps(default_config, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"Warning: Failed to create default MCP config: {e}")

    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            config_data = {
                "version": "1.0",
                "servers": [config.to_dict() for config in self.servers.values()],
            }
            self.config_path.write_text(
                json.dumps(config_data, indent=2),
                encoding="utf-8"
            )
            return True
        except Exception as e:
            print(f"Error saving MCP config: {e}")
            return False

    def add_server(self, config: MCPServerConfig) -> bool:
        """Add an MCP server configuration.
        
        Args:
            config: Server configuration
            
        Returns:
            True if successful
        """
        self.servers[config.name] = config
        return self.save_config()

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration.
        
        Args:
            name: Name of the server to remove
            
        Returns:
            True if successful
        """
        if name in self.servers:
            del self.servers[name]
            return self.save_config()
        return False

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get server configuration by name.
        
        Args:
            name: Server name
            
        Returns:
            Server configuration if found
        """
        return self.servers.get(name)

    def list_servers(self) -> list[MCPServerConfig]:
        """List all configured servers.
        
        Returns:
            List of server configurations
        """
        return list(self.servers.values())

    def enable_server(self, name: str) -> bool:
        """Enable a server.
        
        Args:
            name: Server name
            
        Returns:
            True if successful
        """
        if name in self.servers:
            self.servers[name].enabled = True
            return self.save_config()
        return False

    def disable_server(self, name: str) -> bool:
        """Disable a server.
        
        Args:
            name: Server name
            
        Returns:
            True if successful
        """
        if name in self.servers:
            self.servers[name].enabled = False
            return self.save_config()
        return False

    @asynccontextmanager
    async def create_session(self, server_name: str):
        """Create a session with an MCP server.
        
        Args:
            server_name: Name of the server to connect to
            
        Yields:
            ClientSession instance
        """
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")

        config = self.servers[server_name]
        
        if not config.enabled:
            raise ValueError(f"Server '{server_name}' is disabled")

        if not FASTMCP_AVAILABLE:
            raise RuntimeError("FastMCP is not installed. Run: pip install fastmcp")

        server_params = config.to_server_params()
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self, server_name: str) -> list[dict]:
        """List available tools from an MCP server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of available tools
        """
        async with self.create_session(server_name) as session:
            response = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> Any:
        """Call a tool on an MCP server.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        async with self.create_session(server_name) as session:
            response = await session.call_tool(tool_name, arguments)
            return response

    async def list_resources(self, server_name: str) -> list[dict]:
        """List available resources from an MCP server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of available resources
        """
        async with self.create_session(server_name) as session:
            response = await session.list_resources()
            return [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": resource.mimeType,
                }
                for resource in response.resources
            ]

    async def read_resource(self, server_name: str, resource_uri: str) -> Any:
        """Read a resource from an MCP server.
        
        Args:
            server_name: Name of the server
            resource_uri: URI of the resource to read
            
        Returns:
            Resource content
        """
        async with self.create_session(server_name) as session:
            response = await session.read_resource(resource_uri)
            return response

    def get_status(self) -> dict:
        """Get MCP client status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.enabled,
            "fastmcp_available": FASTMCP_AVAILABLE,
            "config_path": str(self.config_path),
            "servers_count": len(self.servers),
            "servers": [
                {
                    "name": config.name,
                    "command": f"{config.command} {' '.join(config.args)}",
                    "enabled": config.enabled,
                    "description": config.description,
                }
                for config in self.servers.values()
            ],
        }


class MCPManager:
    """High-level manager for MCP integration in IdeaAgent."""

    def __init__(self, client: Optional[MCPClient] = None):
        """Initialize MCP manager.
        
        Args:
            client: Optional MCPClient instance
        """
        self.client = client or MCPClient()

    def is_available(self) -> bool:
        """Check if MCP is available and enabled."""
        return self.client.enabled and len(self.client.servers) > 0

    def get_available_tools_sync(self) -> list[dict]:
        """Get all available tools from enabled servers (synchronous wrapper).
        
        Returns:
            List of available tools with server information
        """
        if not self.is_available():
            return []

        try:
            return asyncio.run(self.get_available_tools())
        except Exception as e:
            print(f"Warning: Failed to get MCP tools: {e}")
            return []

    async def get_available_tools(self) -> list[dict]:
        """Get all available tools from enabled servers.
        
        Returns:
            List of available tools with server information
        """
        all_tools = []
        
        for config in self.client.servers.values():
            if config.enabled:
                try:
                    tools = await self.client.list_tools(config.name)
                    for tool in tools:
                        tool["server"] = config.name
                        all_tools.append(tool)
                except Exception as e:
                    print(f"Warning: Failed to get tools from {config.name}: {e}")
        
        return all_tools

    def format_tools_for_prompt(self) -> str:
        """Format available tools for LLM prompt.
        
        Returns:
            Formatted string describing available tools
        """
        tools = self.get_available_tools_sync()
        
        if not tools:
            return "<mcp_tools>\nNo MCP tools available.\n</mcp_tools>"

        lines = ["<mcp_tools>"]
        for tool in tools:
            lines.append("<tool>")
            lines.append(f"<name>{tool.get('name', 'unknown')}</name>")
            lines.append(f"<description>{tool.get('description', '')}</description>")
            lines.append(f"<server>{tool.get('server', 'unknown')}</server>")
            
            # Include input schema if available
            if "input_schema" in tool:
                lines.append(f"<input_schema>{json.dumps(tool['input_schema'])}</input_schema>")
            
            lines.append("</tool>")
        lines.append("</mcp_tools>")

        return "\n".join(lines)

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str],
        description: str = "",
        env: Optional[dict] = None,
    ) -> bool:
        """Add a new MCP server.
        
        Args:
            name: Server name
            command: Command to run
            args: Command arguments
            description: Server description
            env: Environment variables
            
        Returns:
            True if successful
        """
        config = MCPServerConfig(
            name=name,
            command=command,
            args=args,
            description=description,
            env=env or {},
        )
        return self.client.add_server(config)

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if successful
        """
        return self.client.remove_server(name)


# Example usage as a context manager
@asynccontextmanager
async def mcp_context(server_name: str):
    """Context manager for MCP session.
    
    Args:
        server_name: Name of the server to connect to
        
    Yields:
        MCPClient instance with active session
        
    Example:
        async with mcp_context("filesystem") as client:
            tools = await client.list_tools("filesystem")
    """
    client = MCPClient()
    try:
        yield client
    finally:
        pass  # Cleanup if needed
