from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import atexit
import json
import os
import queue
import shlex
import time
import subprocess
import threading
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass
class VehicleCommandResult:
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseVehicleAdapter(ABC):
    """车控适配层抽象。"""

    @abstractmethod
    def vehicle_climate(
        self,
        op: str = "status",
        target_temp: Optional[int] = None,
        delta: Optional[int] = None,
        fan_speed: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> VehicleCommandResult:
        raise NotImplementedError

    @abstractmethod
    def vehicle_window(self, op: str = "status", position: str = "all", percent: Optional[int] = None) -> VehicleCommandResult:
        raise NotImplementedError

    @abstractmethod
    def vehicle_seat(
        self,
        op: str = "status",
        position: str = "driver",
        level: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> VehicleCommandResult:
        raise NotImplementedError

    @abstractmethod
    def vehicle_navigation(self, destination: str, waypoint: str = "", mode: str = "drive") -> VehicleCommandResult:
        raise NotImplementedError

    @abstractmethod
    def vehicle_media(
        self,
        op: str = "play",
        source: Optional[str] = None,
        track: Optional[str] = None,
        volume: Optional[int] = None,
    ) -> VehicleCommandResult:
        raise NotImplementedError

    @abstractmethod
    def vehicle_status(self) -> VehicleCommandResult:
        raise NotImplementedError

    def invoke_command(self, command_name: str, payload: Dict[str, Any]) -> VehicleCommandResult:
        raise NotImplementedError


class MockVehicleBus(BaseVehicleAdapter):
    """模拟车控总线，用于当前项目落地和联调。"""

    COMMAND_ALIASES = {
        "climate.set": "vehicle_climate",
        "climate.set_temperature": "vehicle_climate",
        "climate.adjust_temperature": "vehicle_climate",
        "climate.set_fan_speed": "vehicle_climate",
        "climate.set_mode": "vehicle_climate",
        "climate.query_status": "vehicle_climate",
        "window.set": "vehicle_window",
        "window.open": "vehicle_window",
        "window.close": "vehicle_window",
        "window.set_position": "vehicle_window",
        "window.query_status": "vehicle_window",
        "seat.set": "vehicle_seat",
        "seat.set_heating": "vehicle_seat",
        "seat.set_cooling": "vehicle_seat",
        "seat.set_massage": "vehicle_seat",
        "seat.stop_massage": "vehicle_seat",
        "seat.adjust_position": "vehicle_seat",
        "seat.query_status": "vehicle_seat",
        "navigation.route": "vehicle_navigation",
        "navigation.navigate_to": "vehicle_navigation",
        "navigation.set_waypoint": "vehicle_navigation",
        "navigation.cancel": "vehicle_navigation",
        "navigation.query_status": "vehicle_navigation",
        "media.control": "vehicle_media",
        "media.play": "vehicle_media",
        "media.pause": "vehicle_media",
        "media.next": "vehicle_media",
        "media.prev": "vehicle_media",
        "media.set_volume": "vehicle_media",
        "media.set_source": "vehicle_media",
        "media.query_status": "vehicle_media",
        "vehicle.status": "vehicle_status",
        "vehicle.query_status": "vehicle_status",
    }

    def __init__(self):
        self.climate = {
            "temperature": 22,
            "fan_speed": 3,
            "mode": "auto",
            "power": True,
        }
        self.windows = {
            "all": 0,
            "front_left": 0,
            "front_right": 0,
            "rear_left": 0,
            "rear_right": 0,
            "sunroof": 0,
        }
        self.seats = {
            "driver": {"heat": 0, "cool": 0, "massage": False, "position": "neutral"},
            "passenger": {"heat": 0, "cool": 0, "massage": False, "position": "neutral"},
        }
        self.media = {
            "playing": False,
            "volume": 18,
            "source": "local",
            "track": "",
        }
        self.navigation = {
            "destination": "",
            "waypoint": "",
            "mode": "drive",
        }
        self.status = {
            "tire_pressure": "normal",
            "range_km": 420,
            "fuel_percent": 58,
            "battery_percent": 76,
            "maintenance": "normal",
        }

    def vehicle_climate(
        self,
        op: str = "status",
        target_temp: Optional[int] = None,
        delta: Optional[int] = None,
        fan_speed: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> VehicleCommandResult:
        if mode:
            self.climate["mode"] = mode
        if fan_speed is not None:
            self.climate["fan_speed"] = max(1, min(7, int(fan_speed)))
        if target_temp is not None:
            self.climate["temperature"] = max(16, min(30, int(target_temp)))
        elif delta:
            self.climate["temperature"] = max(16, min(30, self.climate["temperature"] + int(delta)))
        else:
            if op in ("temp_up", "up"):
                self.climate["temperature"] = min(30, self.climate["temperature"] + 1)
            elif op in ("temp_down", "down"):
                self.climate["temperature"] = max(16, self.climate["temperature"] - 1)

        return VehicleCommandResult(
            success=True,
            message=f"已将空调设置为 {self.climate['temperature']} 度，风量 {self.climate['fan_speed']} 档。",
            data={"climate": dict(self.climate)},
        )

    def vehicle_window(self, op: str = "status", position: str = "all", percent: Optional[int] = None) -> VehicleCommandResult:
        if op in ("status", "query", "query_status"):
            return VehicleCommandResult(
                success=True,
                message=f"车窗状态：{self.windows}",
                data={"windows": dict(self.windows)},
            )

        if op in ("set_position", "set", "move_to"):
            if percent is not None:
                value = max(0, min(100, int(percent)))
            else:
                value = self.windows.get(position, self.windows["all"])
        else:
            value = 0 if op in ("close", "down", "lower") else 100
            if percent is not None:
                value = max(0, min(100, int(percent)))

        if position == "all":
            for key in self.windows:
                self.windows[key] = value
        else:
            if position not in self.windows:
                position = "all"
                for key in self.windows:
                    self.windows[key] = value
            else:
                self.windows[position] = value

        return VehicleCommandResult(
            success=True,
            message=f"已将{position}车窗调整到 {value}%。",
            data={"windows": dict(self.windows)},
        )

    def vehicle_seat(
        self,
        op: str = "status",
        position: str = "driver",
        level: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> VehicleCommandResult:
        seat = self.seats.get(position, self.seats["driver"])
        if op in ("heat_on", "heat", "seat_heat"):
            seat["heat"] = max(1, int(level or 1))
            seat["cool"] = 0
        elif op in ("cool_on", "cool", "seat_cool"):
            seat["cool"] = max(1, int(level or 1))
            seat["heat"] = 0
        elif op in ("massage_on", "massage"):
            seat["massage"] = True
        elif op in ("massage_off", "stop_massage"):
            seat["massage"] = False
        elif op in ("forward", "backward", "forward_adjust", "back_adjust"):
            seat["position"] = direction or op

        self.seats[position] = seat
        return VehicleCommandResult(
            success=True,
            message=f"已调整{position}座椅状态。",
            data={"seats": dict(self.seats)},
        )

    def vehicle_navigation(self, destination: str, waypoint: str = "", mode: str = "drive") -> VehicleCommandResult:
        self.navigation["destination"] = destination
        self.navigation["waypoint"] = waypoint
        self.navigation["mode"] = mode
        return VehicleCommandResult(
            success=True,
            message=f"已开始导航到 {destination}。",
            data={"navigation": dict(self.navigation)},
        )

    def vehicle_media(
        self,
        op: str = "play",
        source: Optional[str] = None,
        track: Optional[str] = None,
        volume: Optional[int] = None,
    ) -> VehicleCommandResult:
        if op in ("set_volume", "volume"):
            if volume is not None:
                self.media["volume"] = max(0, min(30, int(volume)))
            return VehicleCommandResult(
                success=True,
                message=f"已将音量调整到 {self.media['volume']}。",
                data={"media": dict(self.media)},
            )

        if op in ("set_source",):
            if source:
                self.media["source"] = source
            return VehicleCommandResult(
                success=True,
                message=f"已将媒体来源切换为 {self.media['source']}。",
                data={"media": dict(self.media)},
            )

        if op in ("status", "query", "query_status"):
            return VehicleCommandResult(
                success=True,
                message=f"媒体状态：{self.media}",
                data={"media": dict(self.media)},
            )

        if source:
            self.media["source"] = source
        if track:
            self.media["track"] = track
        if volume is not None:
            self.media["volume"] = max(0, min(30, int(volume)))

        if op in ("play", "resume"):
            self.media["playing"] = True
        elif op in ("pause", "stop"):
            self.media["playing"] = False
        elif op in ("next", "next_track"):
            self.media["track"] = "下一首"
            self.media["playing"] = True
        elif op in ("prev", "previous_track"):
            self.media["track"] = "上一首"
            self.media["playing"] = True

        return VehicleCommandResult(
            success=True,
            message=f"已执行媒体操作 {op}。当前音量 {self.media['volume']}。",
            data={"media": dict(self.media)},
        )

    def vehicle_status(self) -> VehicleCommandResult:
        summary = (
            f"胎压{self.status['tire_pressure']}，续航{self.status['range_km']}公里，"
            f"油量{self.status['fuel_percent']}%，电量{self.status['battery_percent']}%，"
            f"保养状态{self.status['maintenance']}。"
        )
        return VehicleCommandResult(success=True, message=summary, data={"status": dict(self.status)})

    def invoke_command(self, command_name: str, payload: Dict[str, Any]) -> VehicleCommandResult:
        try:
            return self._execute_atomic_command(command_name, payload)
        except Exception as exc:
            return VehicleCommandResult(False, f"模拟车控命令执行失败: {exc}", error="invoke_failed")

    def _execute_atomic_command(self, command_name: str, payload: Dict[str, Any]) -> VehicleCommandResult:
        payload = payload or {}

        if command_name in {"climate.set_temperature", "climate.adjust_temperature", "climate.set_fan_speed", "climate.set_mode", "climate.query_status"}:
            if command_name == "climate.query_status":
                return self.vehicle_climate(op="status")
            if command_name == "climate.set_temperature":
                return self.vehicle_climate(op="set_temp", target_temp=payload.get("target_temperature"))
            if command_name == "climate.adjust_temperature":
                delta = payload.get("temperature_delta")
                if delta is None:
                    delta = 1 if payload.get("operation") in ("temp_up", "up", "increase") else -1
                return self.vehicle_climate(op="temp_up" if int(delta) >= 0 else "temp_down", delta=int(delta))
            if command_name == "climate.set_fan_speed":
                return self.vehicle_climate(op="status", fan_speed=payload.get("fan_speed"))
            if command_name == "climate.set_mode":
                return self.vehicle_climate(op="status", mode=payload.get("mode"))

        if command_name in {"window.open", "window.close", "window.set_position", "window.query_status"}:
            if command_name == "window.query_status":
                return self.vehicle_window(op="status")
            op = "open" if command_name == "window.open" else "close" if command_name == "window.close" else "set_position"
            return self.vehicle_window(op=op, position=payload.get("position", "all"), percent=payload.get("percent"))

        if command_name in {"seat.set_heating", "seat.set_cooling", "seat.set_massage", "seat.stop_massage", "seat.adjust_position", "seat.query_status"}:
            if command_name == "seat.query_status":
                return self.vehicle_seat(op="status")
            if command_name == "seat.set_heating":
                return self.vehicle_seat(op="heat_on", position=payload.get("position", "driver"), level=payload.get("level"))
            if command_name == "seat.set_cooling":
                return self.vehicle_seat(op="cool_on", position=payload.get("position", "driver"), level=payload.get("level"))
            if command_name == "seat.set_massage":
                return self.vehicle_seat(op="massage_on", position=payload.get("position", "driver"), level=payload.get("level"))
            if command_name == "seat.stop_massage":
                return self.vehicle_seat(op="massage_off", position=payload.get("position", "driver"))
            if command_name == "seat.adjust_position":
                direction = payload.get("direction") or payload.get("operation") or "forward"
                return self.vehicle_seat(op=direction, position=payload.get("position", "driver"), direction=direction)

        if command_name in {"navigation.navigate_to", "navigation.set_waypoint", "navigation.cancel", "navigation.query_status"}:
            if command_name == "navigation.query_status":
                return VehicleCommandResult(
                    success=True,
                    message=f"导航状态：{self.navigation}",
                    data={"navigation": dict(self.navigation)},
                )
            if command_name == "navigation.cancel":
                self.navigation["destination"] = ""
                self.navigation["waypoint"] = ""
                return VehicleCommandResult(success=True, message="已取消导航。", data={"navigation": dict(self.navigation)})
            if command_name == "navigation.set_waypoint":
                self.navigation["waypoint"] = payload.get("waypoint", "")
                return VehicleCommandResult(success=True, message=f"已设置途经点 {self.navigation['waypoint']}。", data={"navigation": dict(self.navigation)})
            return self.vehicle_navigation(destination=payload.get("destination", ""), waypoint=payload.get("waypoint", ""), mode=payload.get("mode", "drive"))

        if command_name in {"media.play", "media.pause", "media.next", "media.prev", "media.set_volume", "media.set_source", "media.query_status"}:
            if command_name == "media.query_status":
                return self.vehicle_media(op="status")
            if command_name == "media.play":
                return self.vehicle_media(op="play", source=payload.get("source"), track=payload.get("track"), volume=payload.get("volume"))
            if command_name == "media.pause":
                return self.vehicle_media(op="pause", source=payload.get("source"), track=payload.get("track"), volume=payload.get("volume"))
            if command_name == "media.next":
                return self.vehicle_media(op="next", source=payload.get("source"), track=payload.get("track"), volume=payload.get("volume"))
            if command_name == "media.prev":
                return self.vehicle_media(op="prev", source=payload.get("source"), track=payload.get("track"), volume=payload.get("volume"))
            if command_name == "media.set_volume":
                return self.vehicle_media(op="set_volume", volume=payload.get("volume"))
            if command_name == "media.set_source":
                return self.vehicle_media(op="set_source", source=payload.get("source"))

        if command_name in {"vehicle.status", "vehicle.query_status"}:
            return self.vehicle_status()

        normalized_name, normalized_payload = _normalize_vehicle_protocol_command(command_name, payload)
        handler = getattr(self, normalized_name, None)
        if handler is None:
            return VehicleCommandResult(False, f"模拟车控不支持命令: {command_name}", error="command_not_found")
        return handler(**normalized_payload)


class HttpVehicleBusAdapter(BaseVehicleAdapter):
    """通过 HTTP 或 JSON-RPC 对接真实车控服务。"""

    def __init__(
        self,
        base_url: str,
        *,
        protocol: str = "rest",
        endpoint: str = "/vehicle/tools/invoke",
        timeout: float = 5.0,
        auth_token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.protocol = (protocol or "rest").strip().lower()
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.timeout = timeout
        self.auth_token = auth_token

    def vehicle_climate(
        self,
        op: str = "status",
        target_temp: Optional[int] = None,
        delta: Optional[int] = None,
        fan_speed: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> VehicleCommandResult:
        return self._invoke("vehicle_climate", {
            "op": op,
            "target_temp": target_temp,
            "delta": delta,
            "fan_speed": fan_speed,
            "mode": mode,
        })

    def vehicle_window(self, op: str = "status", position: str = "all", percent: Optional[int] = None) -> VehicleCommandResult:
        return self._invoke("vehicle_window", {"op": op, "position": position, "percent": percent})

    def vehicle_seat(
        self,
        op: str = "status",
        position: str = "driver",
        level: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> VehicleCommandResult:
        return self._invoke("vehicle_seat", {"op": op, "position": position, "level": level, "direction": direction})

    def vehicle_navigation(self, destination: str, waypoint: str = "", mode: str = "drive") -> VehicleCommandResult:
        return self._invoke("vehicle_navigation", {"destination": destination, "waypoint": waypoint, "mode": mode})

    def vehicle_media(
        self,
        op: str = "play",
        source: Optional[str] = None,
        track: Optional[str] = None,
        volume: Optional[int] = None,
    ) -> VehicleCommandResult:
        return self._invoke("vehicle_media", {"op": op, "source": source, "track": track, "volume": volume})

    def vehicle_status(self) -> VehicleCommandResult:
        return self._invoke("vehicle_status", {})

    def invoke_command(self, command_name: str, payload: Dict[str, Any]) -> VehicleCommandResult:
        cleaned_payload = {k: v for k, v in (payload or {}).items() if v is not None}
        return self._invoke(command_name, cleaned_payload)

    def _invoke(self, tool_name: str, payload: Dict[str, Any]) -> VehicleCommandResult:
        body = self._build_body(tool_name, payload)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        request = urllib_request.Request(
            self.base_url + self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return self._parse_response(raw, tool_name)
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            return VehicleCommandResult(False, f"车控服务 HTTP 错误: {exc.code}", error=raw)
        except urllib_error.URLError as exc:
            return VehicleCommandResult(False, f"无法连接真实车控服务: {exc.reason}", error="connection_failed")
        except Exception as exc:  # pragma: no cover - defensive guard
            return VehicleCommandResult(False, f"调用真实车控服务失败: {exc}", error="invoke_failed")

    def _build_body(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.protocol == "jsonrpc":
            return {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": tool_name,
                "params": payload,
            }
        return {
            "tool": tool_name,
            "arguments": payload,
        }

    def _parse_response(self, raw: str, tool_name: str) -> VehicleCommandResult:
        try:
            data = json.loads(raw)
        except Exception:
            return VehicleCommandResult(False, raw[:300] or f"真实车控服务返回了非 JSON 响应: {tool_name}", error="invalid_response")

        if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
            data = data["result"]

        if isinstance(data, dict):
            if "success" in data or "message" in data:
                return VehicleCommandResult(
                    success=bool(data.get("success", True)),
                    message=str(data.get("message", "")),
                    data=data.get("data", {}) if isinstance(data.get("data", {}), dict) else {"raw": data.get("data")},
                    error=str(data.get("error", "")),
                )

            if "error" in data:
                error_block = data["error"]
                if isinstance(error_block, dict):
                    return VehicleCommandResult(False, str(error_block.get("message", "车控服务返回错误")), error=str(error_block))
                return VehicleCommandResult(False, str(error_block), error=str(error_block))

            return VehicleCommandResult(True, "车控服务执行成功。", data=data)

        return VehicleCommandResult(True, str(data), data={"raw": data})


class StdioJsonRpcTransport:
    """最小可用的 MCP stdio 传输层。

    采用 Content-Length framing 发送和接收 JSON-RPC 消息，适用于本地启动的 MCP server。
    """

    def __init__(self, command: list[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        if not command:
            raise ValueError("MCP command is required")

        self.command = command
        self.cwd = cwd or os.getcwd()
        self.env = env or os.environ.copy()
        self.process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Failed to start MCP stdio process")

        self._stdin = self.process.stdin
        self._stdout = self.process.stdout
        self._stderr = self.process.stderr
        self._write_lock = threading.Lock()
        self._pending: Dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._message_id = 0
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        atexit.register(self.close)

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
        request_id = self._next_id()
        response_queue: queue.Queue = queue.Queue(maxsize=1)

        with self._pending_lock:
            self._pending[request_id] = response_queue

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self._send(message)

        try:
            response = response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"MCP request timed out: {method}") from exc

        if isinstance(response, dict) and "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise RuntimeError(error.get("message") or str(error))
            raise RuntimeError(str(error))

        return response

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        try:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                except Exception:
                    pass
        finally:
            try:
                if self._stdin:
                    self._stdin.close()
            except Exception:
                pass
            try:
                if self._stdout:
                    self._stdout.close()
            except Exception:
                pass
            try:
                if self._stderr:
                    self._stderr.close()
            except Exception:
                pass

    def _next_id(self) -> int:
        self._message_id += 1
        return self._message_id

    def _send(self, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        framed = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw
        with self._write_lock:
            self._stdin.write(framed)
            self._stdin.flush()

    def _read_loop(self) -> None:
        buffer = b""
        while True:
            try:
                chunk = self._stdout.read(4096)
            except Exception:
                break

            if not chunk:
                break
            buffer += chunk

            while True:
                message, buffer = self._extract_message(buffer)
                if message is None:
                    break
                request_id = message.get("id")
                if request_id is None:
                    continue
                with self._pending_lock:
                    response_queue = self._pending.pop(request_id, None)
                if response_queue is not None:
                    response_queue.put(message)

    def _stderr_loop(self) -> None:
        if not self._stderr:
            return
        while True:
            try:
                chunk = self._stderr.readline()
            except Exception:
                break
            if not chunk:
                break
            try:
                text = chunk.decode("utf-8", errors="replace").rstrip()
                if text:
                    print(f"[MCP STDERR] {text}")
            except Exception:
                pass

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


class MCPStdioVehicleAdapter(BaseVehicleAdapter):
    """真正的 MCP 车控适配器，使用 stdio JSON-RPC 调用本地或远程 MCP server。"""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        protocol_version: str = "2024-11-05",
        client_name: str = "ASR-LLM-TTS",
        client_version: str = "1.0.0",
        tool_timeout: float = 10.0,
        validate_tools: bool = True,
    ):
        self.transport = StdioJsonRpcTransport(command=command, cwd=cwd, env=env)
        self.protocol_version = protocol_version
        self.client_name = client_name
        self.client_version = client_version
        self.tool_timeout = tool_timeout
        self.available_tools: set[str] = set()
        self._initialize()
        if validate_tools:
            self._refresh_tools()

    def _initialize(self) -> None:
        self.transport.request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {},
                    "experimental": {},
                },
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
            timeout=self.tool_timeout,
        )
        try:
            self.transport.notify("notifications/initialized", {})
        except Exception:
            pass

    def _refresh_tools(self) -> None:
        try:
            response = self.transport.request("tools/list", {}, timeout=self.tool_timeout)
            payload = response.get("result", response) if isinstance(response, dict) else {}
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            self.available_tools = {
                tool.get("name") for tool in tools if isinstance(tool, dict) and tool.get("name")
            }
        except Exception as exc:
            print(f"[MCP] tools/list 失败，将继续按调用时校验: {exc}")

    def vehicle_climate(
        self,
        op: str = "status",
        target_temp: Optional[int] = None,
        delta: Optional[int] = None,
        fan_speed: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> VehicleCommandResult:
        return self._call_tool("vehicle_climate", {
            "op": op,
            "target_temp": target_temp,
            "delta": delta,
            "fan_speed": fan_speed,
            "mode": mode,
        })

    def vehicle_window(self, op: str = "status", position: str = "all", percent: Optional[int] = None) -> VehicleCommandResult:
        return self._call_tool("vehicle_window", {"op": op, "position": position, "percent": percent})

    def vehicle_seat(
        self,
        op: str = "status",
        position: str = "driver",
        level: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> VehicleCommandResult:
        return self._call_tool("vehicle_seat", {"op": op, "position": position, "level": level, "direction": direction})

    def vehicle_navigation(self, destination: str, waypoint: str = "", mode: str = "drive") -> VehicleCommandResult:
        return self._call_tool("vehicle_navigation", {"destination": destination, "waypoint": waypoint, "mode": mode})

    def vehicle_media(
        self,
        op: str = "play",
        source: Optional[str] = None,
        track: Optional[str] = None,
        volume: Optional[int] = None,
    ) -> VehicleCommandResult:
        return self._call_tool("vehicle_media", {"op": op, "source": source, "track": track, "volume": volume})

    def vehicle_status(self) -> VehicleCommandResult:
        return self._call_tool("vehicle_status", {})

    def invoke_command(self, command_name: str, payload: Dict[str, Any]) -> VehicleCommandResult:
        return self._call_tool(command_name, payload)

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> VehicleCommandResult:
        if self.available_tools and tool_name not in self.available_tools:
            return VehicleCommandResult(False, f"MCP server 不暴露工具: {tool_name}", error="tool_not_exposed")

        try:
            response = self.transport.request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": {k: v for k, v in arguments.items() if v is not None},
                },
                timeout=self.tool_timeout,
            )
        except Exception as exc:
            return VehicleCommandResult(False, f"MCP 调用失败: {exc}", error="mcp_call_failed")

        payload = response.get("result", response) if isinstance(response, dict) else response
        return self._convert_result(payload, tool_name)

    def _convert_result(self, response: Dict[str, Any], tool_name: str) -> VehicleCommandResult:
        if not isinstance(response, dict):
            return VehicleCommandResult(True, str(response), data={"raw": response})

        is_error = bool(response.get("isError", False))
        content = response.get("content", [])
        structured = response.get("structuredContent")
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "resource", "image"} and item.get("text"):
                    text_parts.append(str(item.get("text")))
                elif item.get("type") == "text" and item.get("text") is not None:
                    text_parts.append(str(item.get("text")))
            elif isinstance(item, str):
                text_parts.append(item)

        message = "\n".join([part for part in text_parts if part]).strip()
        if not message and isinstance(structured, dict):
            message = structured.get("message") or structured.get("summary") or ""
        if not message:
            message = f"MCP 工具 {tool_name} 已执行。"

        data: Dict[str, Any] = {"raw": response}
        if structured is not None:
            data["structuredContent"] = structured

        return VehicleCommandResult(
            success=not is_error,
            message=message,
            data=data,
            error="mcp_tool_error" if is_error else "",
        )


def build_vehicle_adapter() -> BaseVehicleAdapter:
    """根据环境变量选择真实或模拟车控后端。"""

    adapter_kind = (os.environ.get("VEHICLE_ADAPTER") or "mock").strip().lower()
    base_url = (os.environ.get("VEHICLE_API_BASE_URL") or os.environ.get("VEHICLE_MCP_URL") or "").strip()
    protocol = (os.environ.get("VEHICLE_API_PROTOCOL") or "rest").strip().lower()
    endpoint = (os.environ.get("VEHICLE_API_ENDPOINT") or "/vehicle/tools/invoke").strip()
    timeout_raw = os.environ.get("VEHICLE_API_TIMEOUT", "5")
    auth_token = os.environ.get("VEHICLE_API_TOKEN") or os.environ.get("VEHICLE_MCP_TOKEN")
    mcp_command_raw = (os.environ.get("VEHICLE_MCP_COMMAND") or "").strip()
    mcp_args_raw = (os.environ.get("VEHICLE_MCP_ARGS") or "").strip()
    mcp_workdir = (os.environ.get("VEHICLE_MCP_WORKDIR") or "").strip() or None
    mcp_validate_tools = (os.environ.get("VEHICLE_MCP_VALIDATE_TOOLS") or "true").strip().lower() not in {"0", "false", "no"}

    try:
        timeout = float(timeout_raw)
    except Exception:
        timeout = 5.0

    if adapter_kind in {"http", "rest", "remote", "mcp"} and base_url:
        return HttpVehicleBusAdapter(
            base_url,
            protocol=protocol,
            endpoint=endpoint,
            timeout=timeout,
            auth_token=auth_token,
        )

    if adapter_kind in {"mcp-stdio", "mcp_stdio", "stdio"} and mcp_command_raw:
        command = _parse_command_line(mcp_command_raw)
        if mcp_args_raw:
            command.extend(_parse_args_list(mcp_args_raw))
        env = os.environ.copy()
        return MCPStdioVehicleAdapter(
            command=command,
            cwd=mcp_workdir,
            env=env,
            tool_timeout=timeout,
            validate_tools=mcp_validate_tools,
        )

    return MockVehicleBus()


def _parse_command_line(command_raw: str) -> list[str]:
    try:
        if command_raw.startswith("["):
            parsed = json.loads(command_raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass

    if " " in command_raw or "\t" in command_raw:
        return [part for part in shlex.split(command_raw, posix=False) if part]
    return [command_raw]


def _parse_args_list(args_raw: str) -> list[str]:
    try:
        if args_raw.startswith("["):
            parsed = json.loads(args_raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass

    return [part for part in shlex.split(args_raw, posix=False) if part]


def _normalize_vehicle_protocol_command(command_name: str, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    payload = payload or {}
    alias_map = MockVehicleBus.COMMAND_ALIASES
    normalized_name = alias_map.get(command_name, command_name)

    if normalized_name == "vehicle_climate":
        return normalized_name, {
            "op": payload.get("operation") or payload.get("op") or "status",
            "target_temp": payload.get("target_temperature", payload.get("target_temp")),
            "delta": payload.get("temperature_delta", payload.get("delta")),
            "fan_speed": payload.get("fan_speed"),
            "mode": payload.get("mode"),
        }

    if normalized_name == "vehicle_window":
        return normalized_name, {
            "op": payload.get("operation") or payload.get("op") or "status",
            "position": payload.get("position") or "all",
            "percent": payload.get("percent"),
        }

    if normalized_name == "vehicle_seat":
        return normalized_name, {
            "op": payload.get("operation") or payload.get("op") or "status",
            "position": payload.get("position") or "driver",
            "level": payload.get("level"),
            "direction": payload.get("direction"),
        }

    if normalized_name == "vehicle_navigation":
        return normalized_name, {
            "destination": payload.get("destination") or payload.get("target") or "",
            "waypoint": payload.get("waypoint") or "",
            "mode": payload.get("mode") or "drive",
        }

    if normalized_name == "vehicle_media":
        return normalized_name, {
            "op": payload.get("operation") or payload.get("op") or "play",
            "source": payload.get("source"),
            "track": payload.get("track"),
            "volume": payload.get("volume"),
        }

    if normalized_name == "vehicle_status":
        return normalized_name, {}

    return normalized_name, payload
