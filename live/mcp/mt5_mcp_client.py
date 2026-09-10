#!/usr/bin/env python3

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class MCPError(Exception):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPClient:
    """
    Minimal Streamable HTTP MCP client for MetaTrader 5 MCP.

    Flow:
        initialize
        -> notifications/initialized
        -> get_workspace_info (mandatory pre-flight)
        -> tools/list
        -> tools/call
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        protocol_version: str = "2025-06-18",
        client_name: str = "dsh-python-client",
        client_version: str = "1.0",
        timeout: float = 30.0,
        auto_reconnect: bool = True,
    ):
        self.url = url or os.environ.get(
            "MCP_URL",
            "http://127.0.0.1:22346/mcp",
        )

        self.token = token or os.environ.get("MCP_TOKEN")

        self.protocol_version = protocol_version
        self.client_name = client_name
        self.client_version = client_version
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect

        self.session_id: Optional[str] = None
        self.request_id = 0
        self.initialized = False

        self._tools_cache = None

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
        }

        if self.token:
            headers["Authorization"] = "Bearer " + self.token

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        return headers

    @staticmethod
    def _parse_sse(text: str) -> Optional[dict[str, Any]]:
        """
        Parse a simple MCP SSE response.

        Expected:
            data: {"jsonrpc":"2.0",...}

        May also receive multiple data lines.
        """

        data_lines = [
            line[5:].lstrip()
            for line in text.splitlines()
            if line.startswith("data:")
        ]

        if not data_lines:
            return None

        data = "\n".join(data_lines).strip()

        if not data:
            return None

        if data == "[DONE]":
            return None

        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def _decode_response(self, response: Any) -> Optional[dict[str, Any]]:
        """
        Decode either:
            application/json
        or:
            text/event-stream
        """

        content_type = response.headers.get("Content-Type", "").lower()
        text = response.read().decode("utf-8", errors="replace")

        if not text.strip():
            return None

        if "text/event-stream" in content_type:
            result = self._parse_sse(text)

            if result is not None:
                return result

            raise MCPError(
                "MCP returned SSE data that could not be parsed:\n"
                + text
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPError(
                "Invalid JSON returned by MCP server:\n" + text
            ) from exc

    def _post(
        self,
        payload: dict[str, Any],
        *,
        allow_retry: bool = True,
    ) -> Optional[dict[str, Any]]:

        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=body,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                # Session ID can be returned on initialize.
                returned_session = response.headers.get("Mcp-Session-Id")

                if returned_session:
                    self.session_id = returned_session

                return self._decode_response(response)

        except urllib.error.HTTPError as exc:

            response_body = ""
            try:
                response_body = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                pass

            # Session expired / server restarted.
            if (
                exc.code == 400
                and allow_retry
                and self.auto_reconnect
                and self.session_id
            ):
                self.session_id = None
                self.initialized = False
                self._tools_cache = None

                self.connect()

                return self._post(
                    payload,
                    allow_retry=False,
                )

            if exc.code == 401:
                raise MCPConnectionError(
                    "MCP authentication failed (401). "
                    "Check MCP_TOKEN."
                ) from exc

            raise MCPConnectionError(
                f"MCP HTTP {exc.code}: {response_body}"
            ) from exc

        except urllib.error.URLError as exc:
            raise MCPConnectionError(
                f"Cannot connect to MCP server {self.url}: {exc}"
            ) from exc

    # ---------------------------------------------------------
    # MCP lifecycle
    # ---------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """
        Start a new MCP session.
        """

        self.session_id = None
        self.initialized = False
        self._tools_cache = None

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        }

        result = self._post(payload)

        if not result:
            raise MCPError("MCP initialize returned an empty response.")

        if "error" in result:
            raise MCPError(
                "MCP initialize failed: "
                + json.dumps(result["error"], ensure_ascii=False)
            )

        if "result" not in result:
            raise MCPError(
                "Invalid initialize response: "
                + json.dumps(result, ensure_ascii=False)
            )

        # MCP requires this notification after initialize.
        self._post(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        )

        self.initialized = True

        return result["result"]

    def connect(self) -> dict[str, Any]:
        """
        Initialize a fresh MCP session and perform mandatory
        get_workspace_info pre-flight.
        """

        if not self.token:
            raise MCPConnectionError(
                "MCP_TOKEN is not set."
            )

        initialize_result = self.initialize()

        # The MetaTrader 5 MCP server explicitly requires this
        # as the first action in a new session.
        workspace = self.call_tool(
            "get_workspace_info",
            {},
            _skip_preflight=True,
        )
        return {
            "initialize": initialize_result,
            "workspace": workspace,
        }

    # ---------------------------------------------------------
    # MCP tools
    # ---------------------------------------------------------

    def list_tools(self, refresh: bool = False) -> list:
        if not self.initialized:
            self.connect()

        if self._tools_cache is not None and not refresh:
            return self._tools_cache

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }

        response = self._post(payload)

        if not response:
            raise MCPError("tools/list returned empty response.")

        if "error" in response:
            raise MCPError(
                "tools/list failed: "
                + json.dumps(response["error"], ensure_ascii=False)
            )

        result = response.get("result", {})
        tools = result.get("tools", [])

        self._tools_cache = tools

        return tools

    def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
        *,
        _skip_preflight: bool = False,
    ) -> dict[str, Any]:

        if not self.initialized:
            self.connect()

        arguments = arguments or {}

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        response = self._post(payload)

        if not response:
            raise MCPError(
                f"Tool '{name}' returned an empty response."
            )

        if "error" in response:
            error = response["error"]

            # Some MCP servers indicate an invalid/dead session
            # this way.
            message = str(error.get("message", ""))

            if (
                self.auto_reconnect
                and "session" in message.lower()
                and "initialized" in message.lower()
            ):
                self.session_id = None
                self.initialized = False
                self._tools_cache = None

                self.connect()

                return self.call_tool(
                    name,
                    arguments,
                    _skip_preflight=True,
                )

            raise MCPError(
                f"Tool '{name}' failed: "
                + json.dumps(error, ensure_ascii=False)
            )

        return response.get("result", {})

    # ---------------------------------------------------------
    # Convenience functions
    # ---------------------------------------------------------

    def workspace_info(self) -> dict[str, Any]:
        return self.call_tool(
            "get_workspace_info",
            {},
        )

    def close(self):
        """
        MCP Streamable HTTP sessions generally do not require
        an explicit close request.

        Clearing local state guarantees that the next call
        creates a new session.
        """

        self.session_id = None
        self.initialized = False
        self._tools_cache = None

    def health_check(self) -> dict[str, Any]:
        """
        Connect and execute the mandatory workspace pre-flight.
        """

        started = time.time()

        info = self.connect()

        elapsed = time.time() - started

        return {
            "ok": True,
            "url": self.url,
            "session_id": self.session_id,
            "elapsed_seconds": round(elapsed, 3),
            "workspace": info["workspace"],
        }


# =============================================================
# CLI
# =============================================================

def print_json(value: Any):
    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
    )


def main():
    client = MCPClient()

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "\n"
            "  python3 mt5_mcp_client.py health\n"
            "  python3 mt5_mcp_client.py tools\n"
            "  python3 mt5_mcp_client.py workspace\n"
            "  python3 mt5_mcp_client.py call TOOL [JSON_ARGUMENTS]\n"
        )
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "health":
            print_json(client.health_check())

        elif command == "tools":
            client.connect()
            print_json(client.list_tools())

        elif command == "workspace":
            client.connect()
            print_json(client.workspace_info())

        elif command == "call":
            if len(sys.argv) < 3:
                print(
                    "Missing tool name.",
                    file=sys.stderr,
                )
                sys.exit(2)

            tool_name = sys.argv[2]

            arguments = {}

            if len(sys.argv) >= 4:
                arguments = json.loads(sys.argv[3])

            client.connect()

            result = client.call_tool(
                tool_name,
                arguments,
            )

            print_json(result)

        else:
            print(
                f"Unknown command: {command}",
                file=sys.stderr,
            )
            sys.exit(2)

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
