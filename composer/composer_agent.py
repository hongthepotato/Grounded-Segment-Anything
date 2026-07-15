"""LLM composition agent — natural language → block graph via Claude tool-calling.

The agent is given tool definitions that operate on an in-memory graph being built.
Each tool call result is streamed to the browser via SSE.

Tool loop:
1. search_blocks(query) → full block catalog (Claude does semantic matching)
2. get_block_details(block_id) → full block schema
3. validate_connection(...) → alias compatibility check
4. add_block(block_id, params) → adds node to in-progress graph
5. connect(source_node_id, ...) → adds edge (validates first)
6. get_current_graph() → self-inspection
7. finalize_graph() → marks complete, triggers canvas render

Sessions keyed by session_id in memory. TTL: 10 minutes.
Max tool calls per session: 20.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic

from .registry import aliases_compatible, get_block_by_id

logger = logging.getLogger(__name__)

_anthropic_client: Optional[anthropic.AsyncAnthropic] = None

_sessions: Dict[str, Dict] = {}
_SESSION_TTL = timedelta(minutes=10)
_MAX_TOOL_CALLS = 20


def _get_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _make_tools(blocks_catalog: List[Dict]) -> List[Dict]:
    return [
        {
            "name": "search_blocks",
            "description": (
                "Search the block catalog by capability. Returns the full catalog — "
                "perform semantic matching yourself. Use this first to find candidate blocks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language capability description"}
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_block_details",
            "description": "Return the full block definition for a specific block ID.",
            "input_schema": {
                "type": "object",
                "properties": {"block_id": {"type": "string", "description": "Block ID from the catalog"}},
                "required": ["block_id"],
            },
        },
        {
            "name": "validate_connection",
            "description": (
                "Check whether two ports can be connected. Returns valid=true/false with reason. "
                "Always call this BEFORE connect()."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_block_id": {"type": "string"},
                    "source_port_id": {"type": "string"},
                    "target_block_id": {"type": "string"},
                    "target_port_id": {"type": "string"},
                },
                "required": ["source_block_id", "source_port_id", "target_block_id", "target_port_id"],
            },
        },
        {
            "name": "add_block",
            "description": "Add a block to the graph with specified parameter values. Returns node_id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "description": "Block ID to add"},
                    "params": {
                        "type": "object",
                        "description": "Parameter values. Keys must match supervisor_params IDs.",
                    },
                },
                "required": ["block_id"],
            },
        },
        {
            "name": "connect",
            "description": "Connect two block ports. Fails if port aliases are incompatible.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_node_id": {"type": "string"},
                    "source_port_id": {"type": "string"},
                    "target_node_id": {"type": "string"},
                    "target_port_id": {"type": "string"},
                },
                "required": ["source_node_id", "source_port_id", "target_node_id", "target_port_id"],
            },
        },
        {
            "name": "get_current_graph",
            "description": "Return the current in-progress graph state.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "finalize_graph",
            "description": (
                "Mark the graph as complete. This triggers the platform to render it on the canvas "
                "for supervisor review. Call ONLY when the graph is ready."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ]


def _get_port_alias(block: Dict, port_id: str, direction: str) -> Optional[str]:
    ports = block.get("ports", {}).get(direction, [])
    for port in ports:
        if port["id"] == port_id:
            return port["alias"]
    return None


def _execute_tool(tool_name: str, tool_input: Dict, session: Dict, blocks_catalog: List[Dict]) -> Any:
    """Execute a tool and return its result."""
    graph = session["graph"]

    if tool_name == "search_blocks":
        # Return full catalog — Claude does the matching
        return {"blocks": blocks_catalog, "count": len(blocks_catalog)}

    elif tool_name == "get_block_details":
        block_id = tool_input["block_id"]
        block = get_block_by_id(blocks_catalog, block_id)
        if block is None:
            return {"error": f"Block '{block_id}' not found in catalog"}
        return block

    elif tool_name == "validate_connection":
        src_block = get_block_by_id(blocks_catalog, tool_input["source_block_id"])
        tgt_block = get_block_by_id(blocks_catalog, tool_input["target_block_id"])
        if src_block is None:
            return {"valid": False, "reason": f"Source block '{tool_input['source_block_id']}' not found"}
        if tgt_block is None:
            return {"valid": False, "reason": f"Target block '{tool_input['target_block_id']}' not found"}

        src_alias = _get_port_alias(src_block, tool_input["source_port_id"], "output")
        tgt_alias = _get_port_alias(tgt_block, tool_input["target_port_id"], "input")

        if src_alias is None:
            return {
                "valid": False,
                "reason": f"Port '{tool_input['source_port_id']}' not found on {src_block['id']} outputs",
            }
        if tgt_alias is None:
            return {
                "valid": False,
                "reason": f"Port '{tool_input['target_port_id']}' not found on {tgt_block['id']} inputs",
            }

        if aliases_compatible(src_alias, tgt_alias):
            return {"valid": True, "reason": f"Port aliases match: {src_alias} → {tgt_alias}"}
        return {"valid": False, "reason": f"Incompatible aliases: {src_alias} cannot connect to {tgt_alias}"}

    elif tool_name == "add_block":
        block_id = tool_input["block_id"]
        block = get_block_by_id(blocks_catalog, block_id)
        if block is None:
            return {"error": f"Block '{block_id}' not found"}
        node_id = str(uuid.uuid4())
        node = {
            "node_id": node_id,
            "block_id": block_id,
            "block_version": block.get("version", "1.0.0"),
            "params": tool_input.get("params", {}),
            "x": len(graph["nodes"]) * 220,
            "y": 100,
        }
        graph["nodes"].append(node)
        return {"node_id": node_id, "block_id": block_id, "event": "block_added"}

    elif tool_name == "connect":
        source_node_id = tool_input["source_node_id"]
        source_port_id = tool_input["source_port_id"]
        target_node_id = tool_input["target_node_id"]
        target_port_id = tool_input["target_port_id"]

        # Find nodes
        src_node = next((n for n in graph["nodes"] if n["node_id"] == source_node_id), None)
        tgt_node = next((n for n in graph["nodes"] if n["node_id"] == target_node_id), None)
        if src_node is None:
            return {"error": f"Source node {source_node_id} not in graph"}
        if tgt_node is None:
            return {"error": f"Target node {target_node_id} not in graph"}

        # Validate via block catalog
        src_block = get_block_by_id(blocks_catalog, src_node["block_id"])
        tgt_block = get_block_by_id(blocks_catalog, tgt_node["block_id"])
        src_alias = _get_port_alias(src_block, source_port_id, "output") if src_block else None
        tgt_alias = _get_port_alias(tgt_block, target_port_id, "input") if tgt_block else None

        if not aliases_compatible(src_alias or "", tgt_alias or ""):
            return {
                "error": f"Cannot connect: {src_alias} → {tgt_alias} (incompatible aliases)",
                "event": "error",
            }

        edge_id = str(uuid.uuid4())
        edge = {
            "edge_id": edge_id,
            "source_node_id": source_node_id,
            "source_port_id": source_port_id,
            "target_node_id": target_node_id,
            "target_port_id": target_port_id,
        }
        graph["edges"].append(edge)
        return {"edge_id": edge_id, "event": "connected"}

    elif tool_name == "get_current_graph":
        return {"graph": graph}

    elif tool_name == "finalize_graph":
        session["finalized"] = True
        session["graph"]["finalized"] = True
        return {"event": "finalized", "graph": graph}

    return {"error": f"Unknown tool: {tool_name}"}


def create_session(prompt: str, blocks_catalog: List[Dict]) -> str:
    """Create a new composition session. Returns session_id."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "prompt": prompt,
        "graph": {"nodes": [], "edges": []},
        "events": [],
        "finalized": False,
        "created_at": datetime.now(timezone.utc),
        "tool_call_count": 0,
        "blocks_catalog": blocks_catalog,
    }
    logger.info(f"Composition session {session_id} created")
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    session = _sessions.get(session_id)
    if session is None:
        return None
    # Check TTL
    age = datetime.now(timezone.utc) - session["created_at"]
    if age > _SESSION_TTL:
        del _sessions[session_id]
        return None
    return session


async def run_session(session_id: str) -> AsyncIterator[str]:
    """Run the agent loop and yield SSE event strings."""
    session = get_session(session_id)
    if session is None:
        yield _sse_event("error", {"message": "Session not found or expired"})
        return

    blocks_catalog = session["blocks_catalog"]
    tools = _make_tools(blocks_catalog)
    messages = [{"role": "user", "content": session["prompt"]}]
    client = _get_client()

    system_prompt = (
        "You are a ROS2 pipeline composition assistant. The supervisor has described a workflow "
        "in natural language. Your job is to compose it as a block graph using the available tools.\n\n"
        "Rules:\n"
        "1. Always call search_blocks() first to find candidate blocks.\n"
        "2. Always call validate_connection() before connect().\n"
        "3. Call finalize_graph() when the graph is complete — this sends it for supervisor review.\n"
        "4. Never auto-execute. The supervisor must review before Run.\n"
        "5. If you can't find a block for a capability, report it clearly.\n"
        "6. Keep parameter values reasonable — use block defaults unless the supervisor specified otherwise."
    )

    try:
        while session["tool_call_count"] < _MAX_TOOL_CALLS:
            response = await client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            # Collect tool calls from response
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Stream any text narration
            for tb in text_blocks:
                yield _sse_event("message", {"text": tb.text})

            if response.stop_reason == "end_turn" or not tool_calls:
                if not session["finalized"]:
                    yield _sse_event(
                        "warning",
                        {"message": "Composition ended without finalize_graph(). Graph may be incomplete."},
                    )
                break

            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                session["tool_call_count"] += 1
                result = _execute_tool(tool_call.name, tool_call.input, session, blocks_catalog)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

                # Emit SSE event for canvas update
                event_type = result.get("event", tool_call.name)
                yield _sse_event(event_type, result)

                if session["finalized"]:
                    break

            if session["finalized"]:
                break

            # Continue the loop with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        if session["tool_call_count"] >= _MAX_TOOL_CALLS and not session["finalized"]:
            yield _sse_event(
                "warning",
                {
                    "message": f"Reached {_MAX_TOOL_CALLS} tool calls without finishing. "
                    "Partial graph shown — finish manually on the canvas."
                },
            )

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error in session {session_id}: {e}")
        yield _sse_event("error", {"message": f"LLM error: {str(e)}"})

    yield _sse_event("done", {"session_id": session_id, "graph": session["graph"]})


def _sse_event(event_type: str, data: Any) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def check_llm_connectivity() -> bool:
    """Quick check — True if we can reach the Anthropic API."""
    import socket

    try:
        socket.create_connection(("api.anthropic.com", 443), timeout=2)
        return True
    except OSError:
        return False
