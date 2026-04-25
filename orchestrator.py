from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from time import perf_counter


@dataclass
class SkillDispatchResult:
    """技能分发的统一结果。"""

    handled: bool = False
    action: str = ""
    reply: str = ""
    search_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillOrchestrator:
    """根据意图结果调用对应技能，Brain 只处理编排结果。"""

    def __init__(self, skill_registry):
        self.skill_registry = skill_registry

    async def _timed_execute(self, tool_name: str, payload: Dict[str, Any]):
        """统一记录工具调用耗时，供上层写入可观测元数据。"""
        t0 = perf_counter()
        result = await self.skill_registry.execute_tool(tool_name, payload)
        duration_ms = round((perf_counter() - t0) * 1000, 2)
        return result, duration_ms

    async def dispatch(self, intent: Dict[str, Any]) -> SkillDispatchResult:
        climate_action = intent.get("Climate_Action") or {}
        if climate_action:
            reply, duration_ms = await self._timed_execute("vehicle_climate", climate_action)
            return SkillDispatchResult(
                handled=True,
                action="vehicle_climate",
                reply=reply,
                metadata={"intent": climate_action, "tool_name": "vehicle_climate", "tool_duration_ms": duration_ms},
            )

        window_action = intent.get("Window_Action") or {}
        if window_action:
            reply, duration_ms = await self._timed_execute("vehicle_window", window_action)
            return SkillDispatchResult(
                handled=True,
                action="vehicle_window",
                reply=reply,
                metadata={"intent": window_action, "tool_name": "vehicle_window", "tool_duration_ms": duration_ms},
            )

        seat_action = intent.get("Seat_Action") or {}
        if seat_action:
            reply, duration_ms = await self._timed_execute("vehicle_seat", seat_action)
            return SkillDispatchResult(
                handled=True,
                action="vehicle_seat",
                reply=reply,
                metadata={"intent": seat_action, "tool_name": "vehicle_seat", "tool_duration_ms": duration_ms},
            )

        navigation_action = intent.get("Navigation_Action") or {}
        if navigation_action:
            destination = (navigation_action.get("destination") or "").strip() or "目的地"
            reply, duration_ms = await self._timed_execute("vehicle_navigation", navigation_action)
            return SkillDispatchResult(
                handled=True,
                action="vehicle_navigation",
                reply=reply,
                metadata={
                    "intent": navigation_action,
                    "destination": destination,
                    "tool_name": "vehicle_navigation",
                    "tool_duration_ms": duration_ms,
                },
            )

        media_action = intent.get("Media_Action") or {}
        if media_action:
            reply, duration_ms = await self._timed_execute("vehicle_media", media_action)
            return SkillDispatchResult(
                handled=True,
                action="vehicle_media",
                reply=reply,
                metadata={"intent": media_action, "tool_name": "vehicle_media", "tool_duration_ms": duration_ms},
            )

        status_action = intent.get("Vehicle_Status_Action") or {}
        if status_action:
            reply, duration_ms = await self._timed_execute("vehicle_status", status_action)
            return SkillDispatchResult(
                handled=True,
                action="vehicle_status",
                reply=reply,
                metadata={"intent": status_action, "tool_name": "vehicle_status", "tool_duration_ms": duration_ms},
            )

        if intent.get("Call_elm"):
            food_name = (intent.get("Food_candidate") or "").strip() or "随便来点"
            reply, duration_ms = await self._timed_execute("order_food", {"food_name": food_name})
            return SkillDispatchResult(
                handled=True,
                action="order_food",
                reply=reply,
                metadata={"tool_name": "order_food", "tool_duration_ms": duration_ms, "food_name": food_name},
            )

        if intent.get("Need_Search"):
            query = (intent.get("Need_Search") or "").strip()
            if not query:
                return SkillDispatchResult(handled=True, action="web_search", reply="你想查什么呀？")
            search_context, duration_ms = await self._timed_execute("web_search", {"query": query})
            return SkillDispatchResult(
                handled=True,
                action="web_search",
                search_context=search_context,
                metadata={"query": query, "tool_name": "web_search", "tool_duration_ms": duration_ms},
            )

        register_name: Optional[str] = intent.get("Register_Action")
        if register_name:
            reply, duration_ms = await self._timed_execute("register_voice", {"user_name": register_name})
            return SkillDispatchResult(
                handled=True,
                action="register_voice",
                reply=reply,
                metadata={"tool_name": "register_voice", "tool_duration_ms": duration_ms, "user_name": register_name},
            )

        return SkillDispatchResult(handled=False)
    

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    from Knowledge_Grpah import KnowledgeGraph
    from mcp_gateway import MCPGateway
    from skills import SkillRegistry
    import asyncio
    kg = KnowledgeGraph()
    kg.connect()

    skill_registry = SkillRegistry(kg)

    orchestrtor = SkillOrchestrator(skill_registry)

    dict_intent = {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {
                "op": "temp_up",
                "target_temp": 10,
                "delta": 1,
                "fan_speed": 2,
                "mode": "auto",
            },
            "Window_Action": {},
            "Seat_Action": {},
            "Navigation_Action": {},
            "Media_Action": {},
            "Vehicle_Status_Action": {},
    }
    res = asyncio.run(orchestrtor.dispatch(dict_intent))
    print(res)
