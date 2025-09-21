"""Default MCP server configurations used when environment does not specify any.

These definitions provide a simple out-of-the-box setup for the built-in
collectors so that development environments can exercise MCP tooling without
manually wiring environment variables. Adjust the commands, arguments, and
keywords to match the servers available in your workspace.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MCP_SERVERS: list[dict[str, Any]] = [
    {
        "name": "shadcn",
        "command": "npx",
        "args": ["--yes", "shadcn@latest", "mcp"],
        "env": {"npm_config_yes": "true"},
        "trigger_keywords": ["shadcn", "ui", "component"],
    },
    {
        "name": "text-toolkit",
        "command": "npx",
        "args": ["--yes", "@cicatriz/text-toolkit", "mcp"],
        "env": {"npm_config_yes": "true"},
        "trigger_keywords": ["text", "case", "uppercase", "regex"],
    },
]
