from __future__ import annotations

import json
import os
import time
from collections import deque, defaultdict
from dataclasses import asdict
from typing import Any, Dict, Optional

from vehicle_bus import BaseVehicleAdapter, VehicleCommandResult, build_vehicle_adapter


class MCPGateway:
    """面向车控能力的轻量 MCP 网关。

    当前版本以模拟总线为默认后端，保留统一的鉴权、限流、审计入口
    """

    def __init__(self, vehicle_bus: Optional[BaseVehicleAdapter] = None, audit_path: Optional[str] = None):
        self.vehicle_bus = vehicle_bus or build_vehicle_adapter()
        self.audit_path = audit_path or os.path.join(os.getcwd(), "runtime", "mcp_audit.log")
        self.allowed_tools = {
            "vehicle_climate",
            "vehicle_window",
            "vehicle_seat",
            "vehicle_navigation",
            "vehicle_media",
            "vehicle_status",
        }
        self._recent_calls = defaultdict(deque)
        self.rate_limit_window_seconds = 2.0
        self.rate_limit_max_calls = 5

    def invoke(self, tool_name: str, payload: Optional[Dict[str, Any]] = None, actor: str = "system") -> VehicleCommandResult:
        payload = payload or {}
        if tool_name not in self.allowed_tools:
            result = VehicleCommandResult(False, f"未允许的车控工具: {tool_name}", error="tool_not_allowed")
            self._audit(actor, tool_name, payload, result)
            return result

        if not self._allow_call(actor, tool_name):
            result = VehicleCommandResult(False, "操作过于频繁，请稍后再试。", error="rate_limited")
            self._audit(actor, tool_name, payload, result)
            return result

        handler = getattr(self.vehicle_bus, tool_name, None)
        if handler is None:
            result = VehicleCommandResult(False, f"车控总线不支持该工具: {tool_name}", error="tool_missing")
            self._audit(actor, tool_name, payload, result)
            return result

        try:
            result = handler(**payload)
            if not isinstance(result, VehicleCommandResult):
                result = VehicleCommandResult(True, str(result), data={"raw": result})
        except Exception as exc:  # pragma: no cover - defensive guard
            result = VehicleCommandResult(False, f"车控执行失败: {exc}", error="invoke_failed")

        self._audit(actor, tool_name, payload, result)
        return result

    def _allow_call(self, actor: str, tool_name: str) -> bool:
        now = time.time()
        key = f"{actor}:{tool_name}"
        calls = self._recent_calls[key]
        while calls and now - calls[0] > self.rate_limit_window_seconds:
            calls.popleft()
        if len(calls) >= self.rate_limit_max_calls:
            return False
        calls.append(now)
        return True

    def _audit(self, actor: str, tool_name: str, payload: Dict[str, Any], result: VehicleCommandResult) -> None:
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        entry = {
            "ts": time.time(),
            "actor": actor,
            "tool": tool_name,
            "payload": payload,
            "result": asdict(result),
        }
        with open(self.audit_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
