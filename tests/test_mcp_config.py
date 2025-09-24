from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config.mcp_config import McpConfig


def test_single_server_http_fields_are_normalised() -> None:
    config = McpConfig.model_validate(
        {
            "enabled": True,
            "server_transport": "http",
            "server_url": "https://example.com/mcp",
            "server_headers": {"X-Test": "123"},
            "server_timeout": 12.5,
            "server_sse_read_timeout": 4.0,
            "server_terminate_on_close": True,
        }
    )

    assert config.primary_server is not None
    server = config.primary_server
    assert server.transport == "streamable_http"
    assert server.url == "https://example.com/mcp"
    assert server.headers == {"X-Test": "123"}
    assert server.timeout == 12.5
    assert server.sse_read_timeout == 4.0
    assert server.terminate_on_close is True
    assert config.server_transport == "streamable_http"
    assert config.server_url == "https://example.com/mcp"
    assert config.server_headers == {"X-Test": "123"}
    assert config.server_timeout == 12.5
    assert config.server_sse_read_timeout == 4.0
    assert config.server_terminate_on_close is True


def test_single_server_headers_parse_from_json_string() -> None:
    config = McpConfig.model_validate(
        {
            "enabled": True,
            "server_transport": "streamable_http",
            "server_url": "https://example.com/mcp",
            "server_headers": '{"Authorization": "Bearer abc"}',
        }
    )

    server = config.primary_server
    assert server is not None
    assert server.headers == {"Authorization": "Bearer abc"}
    assert config.server_headers == {"Authorization": "Bearer abc"}
