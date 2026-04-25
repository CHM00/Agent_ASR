from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vehicle_bus import BaseVehicleAdapter, HttpVehicleBusAdapter, MockVehicleBus, VehicleCommandResult


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    backend_command: str
    command_map: Dict[str, str] = field(default_factory=dict)
    protocol_commands: Dict[str, "ProtocolCommandSpec"] = field(default_factory=dict)
    argument_map: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolCommandSpec:
    backend_command: str
    payload_map: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)


TOOLS: Dict[str, MCPTool] = {
    "vehicle_climate": MCPTool(
        name="vehicle_climate",
        description="调整空调温度、风量和模式。",
        backend_command="climate.query_status",
        command_map={
            "status": "climate.query_status",
            "query": "climate.query_status",
            "query_status": "climate.query_status",
            "temp_up": "climate.adjust_temperature",
            "temp_down": "climate.adjust_temperature",
            "set_temp": "climate.set_temperature",
            "set_temperature": "climate.set_temperature",
            "set_fan": "climate.set_fan_speed",
            "set_mode": "climate.set_mode",
        },
        protocol_commands={
            "status": ProtocolCommandSpec(backend_command="climate.query_status"),
            "query": ProtocolCommandSpec(backend_command="climate.query_status"),
            "query_status": ProtocolCommandSpec(backend_command="climate.query_status"),
            "temp_up": ProtocolCommandSpec(
                backend_command="climate.adjust_temperature",
                payload_map={"delta": "temperature_delta"},
                defaults={"temperature_delta": 1},
            ),
            "temp_down": ProtocolCommandSpec(
                backend_command="climate.adjust_temperature",
                payload_map={"delta": "temperature_delta"},
                defaults={"temperature_delta": -1},
            ),
            "set_temp": ProtocolCommandSpec(
                backend_command="climate.set_temperature",
                payload_map={"target_temp": "target_temperature"},
            ),
            "set_temperature": ProtocolCommandSpec(
                backend_command="climate.set_temperature",
                payload_map={"target_temp": "target_temperature"},
            ),
            "set_fan": ProtocolCommandSpec(
                backend_command="climate.set_fan_speed",
                payload_map={"fan_speed": "fan_speed"},
            ),
            "set_mode": ProtocolCommandSpec(
                backend_command="climate.set_mode",
                payload_map={"mode": "mode"},
            ),
        },
        input_schema={
            "type": "object",
            "properties": {
                "op": {"type": "string", "description": "操作类型，例如 temp_up、temp_down、status"},
                "target_temp": {"type": "integer", "description": "目标温度"},
                "delta": {"type": "integer", "description": "相对温度调节量"},
                "fan_speed": {"type": "integer", "description": "风量档位"},
                "mode": {"type": "string", "description": "模式，例如 auto、cool、heat"},
            },
            "required": [],
        },
        argument_map={
            "op": "operation",
            "target_temp": "target_temperature",
            "delta": "temperature_delta",
            "fan_speed": "fan_speed",
            "mode": "mode",
        },
        defaults={"operation": "status"},
    ),
    "vehicle_window": MCPTool(
        name="vehicle_window",
        description="控制车窗或天窗开合。",
        backend_command="window.query_status",
        command_map={
            "status": "window.query_status",
            "query": "window.query_status",
            "query_status": "window.query_status",
            "open": "window.open",
            "close": "window.close",
            "set_position": "window.set_position",
        },
        protocol_commands={
            "status": ProtocolCommandSpec(backend_command="window.query_status"),
            "query": ProtocolCommandSpec(backend_command="window.query_status"),
            "query_status": ProtocolCommandSpec(backend_command="window.query_status"),
            "open": ProtocolCommandSpec(
                backend_command="window.open",
                payload_map={"position": "position", "percent": "percent"},
                defaults={"percent": 100},
            ),
            "close": ProtocolCommandSpec(
                backend_command="window.close",
                payload_map={"position": "position", "percent": "percent"},
                defaults={"percent": 0},
            ),
            "set_position": ProtocolCommandSpec(
                backend_command="window.set_position",
                payload_map={"position": "position", "percent": "percent"},
            ),
        },
        input_schema={
            "type": "object",
            "properties": {
                "op": {"type": "string", "description": "操作类型，例如 open、close、status"},
                "position": {"type": "string", "description": "位置，例如 all、front_left、sunroof"},
                "percent": {"type": "integer", "description": "开合百分比"},
            },
            "required": [],
        },
        argument_map={
            "op": "operation",
            "position": "position",
            "percent": "percent",
        },
        defaults={"operation": "status", "position": "all"},
    ),
    "vehicle_seat": MCPTool(
        name="vehicle_seat",
        description="控制座椅加热、通风、按摩和位置。",
        backend_command="seat.query_status",
        command_map={
            "status": "seat.query_status",
            "query": "seat.query_status",
            "query_status": "seat.query_status",
            "heat_on": "seat.set_heating",
            "cool_on": "seat.set_cooling",
            "massage_on": "seat.set_massage",
            "massage_off": "seat.stop_massage",
            "forward": "seat.adjust_position",
            "backward": "seat.adjust_position",
        },
        protocol_commands={
            "status": ProtocolCommandSpec(backend_command="seat.query_status"),
            "query": ProtocolCommandSpec(backend_command="seat.query_status"),
            "query_status": ProtocolCommandSpec(backend_command="seat.query_status"),
            "heat_on": ProtocolCommandSpec(
                backend_command="seat.set_heating",
                payload_map={"position": "position", "level": "level"},
                defaults={"level": 1},
            ),
            "cool_on": ProtocolCommandSpec(
                backend_command="seat.set_cooling",
                payload_map={"position": "position", "level": "level"},
                defaults={"level": 1},
            ),
            "massage_on": ProtocolCommandSpec(
                backend_command="seat.set_massage",
                payload_map={"position": "position", "level": "level"},
                defaults={"level": 1},
            ),
            "massage_off": ProtocolCommandSpec(
                backend_command="seat.stop_massage",
                payload_map={"position": "position"},
            ),
            "forward": ProtocolCommandSpec(
                backend_command="seat.adjust_position",
                payload_map={"position": "position", "direction": "direction"},
                defaults={"direction": "forward"},
            ),
            "backward": ProtocolCommandSpec(
                backend_command="seat.adjust_position",
                payload_map={"position": "position", "direction": "direction"},
                defaults={"direction": "backward"},
            ),
        },
        input_schema={
            "type": "object",
            "properties": {
                "op": {"type": "string", "description": "操作类型，例如 heat_on、cool_on、massage_on、forward"},
                "position": {"type": "string", "description": "座椅位置，例如 driver、passenger"},
                "level": {"type": "integer", "description": "档位"},
                "direction": {"type": "string", "description": "方向，例如 forward、backward"},
            },
            "required": [],
        },
        argument_map={
            "op": "operation",
            "position": "position",
            "level": "level",
            "direction": "direction",
        },
        defaults={"operation": "status", "position": "driver"},
    ),
    "vehicle_navigation": MCPTool(
        name="vehicle_navigation",
        description="发起导航到目的地。",
        backend_command="navigation.query_status",
        command_map={
            "status": "navigation.query_status",
            "query": "navigation.query_status",
            "query_status": "navigation.query_status",
            "route": "navigation.navigate_to",
            "navigate": "navigation.navigate_to",
            "waypoint": "navigation.set_waypoint",
            "cancel": "navigation.cancel",
        },
        protocol_commands={
            "status": ProtocolCommandSpec(backend_command="navigation.query_status"),
            "query": ProtocolCommandSpec(backend_command="navigation.query_status"),
            "query_status": ProtocolCommandSpec(backend_command="navigation.query_status"),
            "route": ProtocolCommandSpec(
                backend_command="navigation.navigate_to",
                payload_map={"destination": "destination", "waypoint": "waypoint", "mode": "mode"},
                defaults={"mode": "drive"},
            ),
            "navigate": ProtocolCommandSpec(
                backend_command="navigation.navigate_to",
                payload_map={"destination": "destination", "waypoint": "waypoint", "mode": "mode"},
                defaults={"mode": "drive"},
            ),
            "waypoint": ProtocolCommandSpec(
                backend_command="navigation.set_waypoint",
                payload_map={"waypoint": "waypoint"},
            ),
            "cancel": ProtocolCommandSpec(backend_command="navigation.cancel"),
        },
        input_schema={
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "目的地"},
                "waypoint": {"type": "string", "description": "途经点"},
                "mode": {"type": "string", "description": "导航模式，例如 drive、walk"},
            },
            "required": ["destination"],
        },
        argument_map={
            "destination": "destination",
            "waypoint": "waypoint",
            "mode": "mode",
        },
        defaults={"mode": "drive"},
    ),
    "vehicle_media": MCPTool(
        name="vehicle_media",
        description="控制车机媒体播放、暂停、切歌和音量。",
        backend_command="media.query_status",
        command_map={
            "status": "media.query_status",
            "query": "media.query_status",
            "query_status": "media.query_status",
            "play": "media.play",
            "pause": "media.pause",
            "next": "media.next",
            "prev": "media.prev",
            "set_volume": "media.set_volume",
            "set_source": "media.set_source",
        },
        protocol_commands={
            "status": ProtocolCommandSpec(backend_command="media.query_status"),
            "query": ProtocolCommandSpec(backend_command="media.query_status"),
            "query_status": ProtocolCommandSpec(backend_command="media.query_status"),
            "play": ProtocolCommandSpec(
                backend_command="media.play",
                payload_map={"source": "source", "track": "track", "volume": "volume"},
            ),
            "pause": ProtocolCommandSpec(
                backend_command="media.pause",
                payload_map={"source": "source", "track": "track", "volume": "volume"},
            ),
            "next": ProtocolCommandSpec(
                backend_command="media.next",
                payload_map={"source": "source", "track": "track", "volume": "volume"},
            ),
            "prev": ProtocolCommandSpec(
                backend_command="media.prev",
                payload_map={"source": "source", "track": "track", "volume": "volume"},
            ),
            "set_volume": ProtocolCommandSpec(
                backend_command="media.set_volume",
                payload_map={"volume": "volume"},
            ),
            "set_source": ProtocolCommandSpec(
                backend_command="media.set_source",
                payload_map={"source": "source"},
            ),
        },
        input_schema={
            "type": "object",
            "properties": {
                "op": {"type": "string", "description": "操作类型，例如 play、pause、next、prev"},
                "source": {"type": "string", "description": "媒体来源"},
                "track": {"type": "string", "description": "曲目或内容"},
                "volume": {"type": "integer", "description": "音量"},
            },
            "required": [],
        },
        argument_map={
            "op": "operation",
            "source": "source",
            "track": "track",
            "volume": "volume",
        },
        defaults={"operation": "play"},
    ),
    "vehicle_status": MCPTool(
        name="vehicle_status",
        description="查询车辆状态。",
        backend_command="vehicle.query_status",
        command_map={
            "status": "vehicle.query_status",
            "query": "vehicle.query_status",
            "query_status": "vehicle.query_status",
        },
        protocol_commands={
            "status": ProtocolCommandSpec(backend_command="vehicle.query_status"),
            "query": ProtocolCommandSpec(backend_command="vehicle.query_status"),
            "query_status": ProtocolCommandSpec(backend_command="vehicle.query_status"),
        },
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        defaults={},
    ),
}

class VehicleMCPService:
    def __init__(self, backend: BaseVehicleAdapter):
        self.backend = backend

    def list_tools(self) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in TOOLS.values()
            ]
        }

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> VehicleCommandResult:
        spec = TOOLS.get(name)
        if not spec:
            return VehicleCommandResult(False, f"未知工具: {name}", error="tool_not_found")

        try:
            action_name, command_name = self._select_backend_command(spec, arguments)
            payload = self._build_backend_payload(spec, arguments, command_name, action_name)
            if hasattr(self.backend, "invoke_command"):
                result = self.backend.invoke_command(command_name, payload)
            else:
                handler = getattr(self.backend, name, None) or getattr(self.backend, spec.backend_command, None)
                if handler is None:
                    return VehicleCommandResult(False, f"后端未实现工具: {name}", error="tool_missing")
                result = handler(**self._build_legacy_payload(name, arguments))
            if isinstance(result, VehicleCommandResult):
                return result
            return VehicleCommandResult(True, str(result), data={"raw": result})
        except Exception as exc:  # pragma: no cover - defensive guard
            return VehicleCommandResult(False, f"工具执行失败: {exc}", error="invoke_failed")

    def _select_backend_command(self, spec: MCPTool, arguments: Dict[str, Any]) -> tuple[str, str]:
        arguments = arguments or {}
        op = str(arguments.get("op") or arguments.get("operation") or "").strip().lower()
        if not op:
            op = spec.defaults.get("operation", "status")

        if spec.name == "vehicle_climate":
            if arguments.get("target_temp") is not None:
                op = "set_temp"
            if arguments.get("delta") is not None or op in {"temp_up", "temp_down", "increase", "decrease"}:
                try:
                    delta = int(arguments.get("delta")) if arguments.get("delta") is not None else None
                except Exception:
                    delta = None
                if op == "temp_down" or (delta is not None and delta < 0):
                    op = "temp_down"
                else:
                    op = "temp_up"
            if arguments.get("fan_speed") is not None:
                op = "set_fan"
            if arguments.get("mode") is not None:
                op = "set_mode"
            command_spec = spec.protocol_commands.get(op)
            return op, (command_spec.backend_command if command_spec else spec.command_map.get(op) or spec.backend_command)

        if spec.name == "vehicle_window":
            if arguments.get("percent") is not None:
                op = "set_position"
            if op in {"open", "close", "set_position"}:
                command_spec = spec.protocol_commands.get(op)
                return op, (command_spec.backend_command if command_spec else spec.command_map.get(op, spec.backend_command))
            command_spec = spec.protocol_commands.get(op)
            return op, (command_spec.backend_command if command_spec else spec.backend_command)

        if spec.name == "vehicle_seat":
            if op in {"heat_on", "cool_on", "massage_on", "massage_off", "forward", "backward"}:
                command_spec = spec.protocol_commands.get(op)
                return op, (command_spec.backend_command if command_spec else spec.command_map.get(op, spec.backend_command))
            command_spec = spec.protocol_commands.get(op)
            return op, (command_spec.backend_command if command_spec else spec.backend_command)

        if spec.name == "vehicle_navigation":
            if arguments.get("waypoint") and not arguments.get("destination"):
                op = "waypoint"
            if op in {"cancel"}:
                command_spec = spec.protocol_commands.get(op)
                return op, (command_spec.backend_command if command_spec else spec.command_map.get(op, spec.backend_command))
            if arguments.get("destination"):
                op = "route"
            command_spec = spec.protocol_commands.get(op)
            return op, (command_spec.backend_command if command_spec else spec.backend_command)

        if spec.name == "vehicle_media":
            if arguments.get("volume") is not None and op in {"set_volume", "volume"}:
                op = "set_volume"
            if arguments.get("source") and not arguments.get("track") and op in {"set_source"}:
                op = "set_source"
            if op in {"play", "pause", "next", "prev", "set_volume", "set_source"}:
                command_spec = spec.protocol_commands.get(op)
                return op, (command_spec.backend_command if command_spec else spec.command_map.get(op, spec.backend_command))
            command_spec = spec.protocol_commands.get(op)
            return op, (command_spec.backend_command if command_spec else spec.backend_command)

        if spec.name == "vehicle_status":
            return op or "status", "vehicle.query_status"

        command_spec = spec.protocol_commands.get(op)
        return op, (command_spec.backend_command if command_spec else spec.backend_command)

    def _build_backend_payload(self, spec: MCPTool, arguments: Dict[str, Any], command_name: str, action_name: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(spec.defaults)
        arguments = arguments or {}
        command_spec = spec.protocol_commands.get(action_name)
        if command_spec is None:
            command_spec = ProtocolCommandSpec(backend_command=command_name)

        payload.update(command_spec.defaults)

        for source_key, target_key in command_spec.payload_map.items():
            value = arguments.get(source_key)
            if value is not None:
                payload[target_key] = value

        if spec.name == "vehicle_navigation" and not payload.get("destination"):
            payload["destination"] = arguments.get("destination") or "目的地"

        payload = {k: v for k, v in payload.items() if v is not None}

        if command_name == "climate.set_temperature":
            return {"target_temperature": payload.get("target_temperature")}
        if command_name == "climate.adjust_temperature":
            return {"temperature_delta": payload.get("temperature_delta") or payload.get("delta") or 1}
        if command_name == "climate.set_fan_speed":
            return {"fan_speed": payload.get("fan_speed")}
        if command_name == "climate.set_mode":
            return {"mode": payload.get("mode")}
        if command_name in {"climate.query_status", "window.query_status", "seat.query_status", "navigation.query_status", "media.query_status", "vehicle.query_status"}:
            return {}

        if command_name == "window.set_position":
            return {"position": payload.get("position", "all"), "percent": payload.get("percent")}
        if command_name == "window.open":
            return {"position": payload.get("position", "all"), "percent": payload.get("percent", 100)}
        if command_name == "window.close":
            return {"position": payload.get("position", "all"), "percent": payload.get("percent", 0)}

        if command_name == "seat.set_heating":
            return {"position": payload.get("position", "driver"), "level": payload.get("level", 1)}
        if command_name == "seat.set_cooling":
            return {"position": payload.get("position", "driver"), "level": payload.get("level", 1)}
        if command_name == "seat.set_massage":
            return {"position": payload.get("position", "driver"), "level": payload.get("level", 1)}
        if command_name == "seat.stop_massage":
            return {"position": payload.get("position", "driver")}
        if command_name == "seat.adjust_position":
            return {"position": payload.get("position", "driver"), "direction": payload.get("direction") or payload.get("op") or "forward"}

        if command_name == "navigation.navigate_to":
            return {"destination": payload.get("destination", "目的地"), "waypoint": payload.get("waypoint", ""), "mode": payload.get("mode", "drive")}
        if command_name == "navigation.set_waypoint":
            return {"waypoint": payload.get("waypoint", "")}
        if command_name == "navigation.cancel":
            return {}

        if command_name == "media.play":
            return {"op": "play", "source": payload.get("source"), "track": payload.get("track"), "volume": payload.get("volume")}
        if command_name == "media.pause":
            return {"op": "pause", "source": payload.get("source"), "track": payload.get("track"), "volume": payload.get("volume")}
        if command_name == "media.next":
            return {"op": "next", "source": payload.get("source"), "track": payload.get("track"), "volume": payload.get("volume")}
        if command_name == "media.prev":
            return {"op": "prev", "source": payload.get("source"), "track": payload.get("track"), "volume": payload.get("volume")}
        if command_name == "media.set_volume":
            return {"volume": payload.get("volume")}
        if command_name == "media.set_source":
            return {"source": payload.get("source")}

        return payload

    def _build_legacy_payload(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
        if tool_name == "vehicle_climate":
            return {
                "op": arguments.get("op", "status"),
                "target_temp": arguments.get("target_temp"),
                "delta": arguments.get("delta"),
                "fan_speed": arguments.get("fan_speed"),
                "mode": arguments.get("mode"),
            }
        if tool_name == "vehicle_window":
            return {
                "op": arguments.get("op", "status"),
                "position": arguments.get("position", "all"),
                "percent": arguments.get("percent"),
            }
        if tool_name == "vehicle_seat":
            return {
                "op": arguments.get("op", "status"),
                "position": arguments.get("position", "driver"),
                "level": arguments.get("level"),
                "direction": arguments.get("direction"),
            }
        if tool_name == "vehicle_navigation":
            return {
                "destination": arguments.get("destination", "目的地"),
                "waypoint": arguments.get("waypoint", ""),
                "mode": arguments.get("mode", "drive"),
            }
        if tool_name == "vehicle_media":
            return {
                "op": arguments.get("op", "play"),
                "source": arguments.get("source"),
                "track": arguments.get("track"),
                "volume": arguments.get("volume"),
            }
        return {}


class StdIOJsonRpcServer:
    def __init__(self, service: VehicleMCPService, server_name: str = "vehicle-mcp-server", server_version: str = "0.1.0"):
        self.service = service
        self.server_name = server_name
        self.server_version = server_version
        self._buffer = b""
        self._message_id = 0

    def serve_forever(self) -> None:
        self._log(f"{self.server_name} started")
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer

        while True:
            chunk = stdin.read(1)
            if not chunk:
                break
            self._buffer += chunk

            while True:
                message, self._buffer = self._extract_message(self._buffer)
                if message is None:
                    break
                response = self._handle_message(message)
                if response is not None:
                    self._write_message(stdout, response)

    def _handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            params = message.get("params", {}) if isinstance(message.get("params", {}), dict) else {}
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                    "serverInfo": {
                        "name": self.server_name,
                        "version": self.server_version,
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": self.service.list_tools(),
            }

        if method == "tools/call":
            params = message.get("params", {}) if isinstance(message.get("params", {}), dict) else {}
            tool_name = params.get("name") or params.get("tool") or ""
            arguments = params.get("arguments", {}) if isinstance(params.get("arguments", {}), dict) else {}
            result = self.service.call_tool(tool_name, arguments)
            payload = self._result_to_mcp(result, tool_name)
            payload["id"] = request_id
            return payload

        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}}

        return self._error(request_id, -32601, f"Unknown method: {method}")

    def _result_to_mcp(self, result: VehicleCommandResult, tool_name: str) -> Dict[str, Any]:
        content_text = result.message or f"MCP 工具 {tool_name} 已执行。"
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{"type": "text", "text": content_text}],
                "structuredContent": {
                    "success": result.success,
                    "message": result.message,
                    "data": result.data,
                    "error": result.error,
                    "tool": tool_name,
                },
                "isError": not result.success,
            },
        }

    def _error(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def _write_message(self, stdout, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
        stdout.write(header + raw)
        stdout.flush()

    def _extract_message(self, buffer: bytes) -> tuple[Optional[Dict[str, Any]], bytes]:
        header_end = buffer.find(b"\r\n\r\n")
        if header_end < 0:
            return None, buffer

        header_blob = buffer[:header_end].decode("ascii", errors="replace")
        content_length = None
        for line in header_blob.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except Exception:
                    content_length = None
                break

        if content_length is None:
            return None, buffer[header_end + 4 :]

        body_start = header_end + 4
        body_end = body_start + content_length
        if len(buffer) < body_end:
            return None, buffer

        body = buffer[body_start:body_end]
        remainder = buffer[body_end:]
        try:
            message = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            return None, remainder
        return message, remainder

    def _log(self, text: str) -> None:
        print(f"[vehicle-mcp-server] {text}", file=sys.stderr, flush=True)


def build_server_backend() -> BaseVehicleAdapter:
    backend_kind = (os.environ.get("VEHICLE_SERVER_BACKEND") or "mock").strip().lower()
    base_url = (os.environ.get("VEHICLE_SERVER_HTTP_BASE_URL") or os.environ.get("VEHICLE_SERVER_BASE_URL") or "").strip()
    protocol = (os.environ.get("VEHICLE_SERVER_HTTP_PROTOCOL") or os.environ.get("VEHICLE_SERVER_PROTOCOL") or "rest").strip().lower()
    endpoint = (os.environ.get("VEHICLE_SERVER_HTTP_ENDPOINT") or "/vehicle/tools/invoke").strip()
    token = os.environ.get("VEHICLE_SERVER_HTTP_TOKEN") or os.environ.get("VEHICLE_SERVER_TOKEN")
    timeout_raw = os.environ.get("VEHICLE_SERVER_HTTP_TIMEOUT", "5")

    try:
        timeout = float(timeout_raw)
    except Exception:
        timeout = 5.0

    if backend_kind in {"mcp", "mcp-stdio", "stdio"}:
        raise ValueError(
            "VEHICLE_SERVER_BACKEND 不能设置为 mcp-stdio；vehicle_mcp_server.py 本身就是 MCP server 进程，"
            "它的 backend 应该指向真实车控 HTTP/JSON-RPC 服务或 mock。"
        )

    if backend_kind == "auto":
        if base_url:
            backend_kind = "http"
        else:
            return MockVehicleBus()

    if backend_kind in {"mock", "local", "stub"}:
        return MockVehicleBus()

    if backend_kind in {"http", "rest", "remote", "jsonrpc"}:
        if not base_url:
            raise ValueError(
                "VEHICLE_SERVER_HTTP_BASE_URL is required when VEHICLE_SERVER_BACKEND is set to a real backend mode."
            )

        if backend_kind == "jsonrpc":
            protocol = "jsonrpc"
        elif backend_kind in {"http", "rest", "remote"} and protocol not in {"rest", "jsonrpc"}:
            protocol = "rest"

        return HttpVehicleBusAdapter(
            base_url=base_url,
            protocol=protocol,
            endpoint=endpoint,
            timeout=timeout,
            auth_token=token,
        )

    raise ValueError(f"不支持的 VEHICLE_SERVER_BACKEND: {backend_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle MCP server over stdio JSON-RPC")
    parser.add_argument("--server-name", default="vehicle-mcp-server")
    parser.add_argument("--server-version", default="0.1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = build_server_backend()
    service = VehicleMCPService(backend)
    server = StdIOJsonRpcServer(service, server_name=args.server_name, server_version=args.server_version)
    server.serve_forever()


if __name__ == "__main__":
    main()
