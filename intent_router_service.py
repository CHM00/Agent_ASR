import asyncio
import json
import re
from typing import Any, Dict, Optional


class IntentRouterService:
    """封装意图路由器，统一输出结构并做兜底。"""

    REQUIRED_KEYS = (
        "Call_elm",
        "Food_candidate",
        "Need_Search",
        "Register_Action",
        "Climate_Action",
        "Window_Action",
        "Seat_Action",
        "Navigation_Action",
        "Media_Action",
        "Vehicle_Status_Action",
    )

    def __init__(
        self,
        router,
        llm_client: Optional[Any] = None,
        llm_model: str = "",
        skill_registry: Optional[Any] = None,
        llm_enabled: bool = True,
        llm_min_confidence: float = 0.55,
    ):
        self.router = router
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.skill_registry = skill_registry
        self.llm_enabled = llm_enabled and llm_client is not None and skill_registry is not None and bool(llm_model)
        self.llm_min_confidence = llm_min_confidence
        self.tool_catalog = skill_registry.get_all_tools() if skill_registry and hasattr(skill_registry, "get_all_tools") else []

    def route(self, text: str) -> Dict[str, Any]:
        return asyncio.run(self.route_async(text))

    async def route_async(self, text: str) -> Dict[str, Any]:
        default = self._build_default_intent()

        if self.llm_enabled:
            try:
                llm_decision = await self._route_with_llm(text)
                if llm_decision:
                    resolved = self._decision_to_intent(text, llm_decision)
                    if resolved:
                        return resolved
            except Exception as e:
                print(f"[IntentRouterService] LLM 路由失败，准备回退: {e}")

        vehicle_intent = self._extract_vehicle_intent(text)
        if vehicle_intent:
            return {**default, **vehicle_intent, "Route_Source": "heuristic"}

        try:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, self.router.route, text)
        except Exception as e:
            print(f"[IntentRouterService] 路由失败，回退chat: {e}")
            return default

        if not isinstance(raw, dict):
            return default

        normalized = default.copy()
        for key in self.REQUIRED_KEYS:
            if key in raw:
                normalized[key] = raw.get(key)

        normalized["Call_elm"] = bool(normalized.get("Call_elm"))
        normalized["Food_candidate"] = str(normalized.get("Food_candidate") or "")
        normalized["Need_Search"] = str(normalized.get("Need_Search") or "")
        normalized["Register_Action"] = str(normalized.get("Register_Action") or "")

        for key in ("Climate_Action", "Window_Action", "Seat_Action", "Navigation_Action", "Media_Action", "Vehicle_Status_Action"):
            if key in raw and isinstance(raw.get(key), dict):
                normalized[key] = raw.get(key)

        normalized["Route_Source"] = "bert"
        return normalized

    def _build_default_intent(self) -> Dict[str, Any]:
        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {},
            "Window_Action": {},
            "Seat_Action": {},
            "Navigation_Action": {},
            "Media_Action": {},
            "Vehicle_Status_Action": {},
            "Need_Clarification": False,
            "Clarification_Prompt": "",
            "Route_Source": "default",
            "Route_Confidence": 0.0,
        }

    async def _route_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        if not text.strip() or not self.tool_catalog:
            return None

        prompt = self._build_llm_prompt(text)
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=prompt,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None

        return self._parse_json_object(content)

    def _build_llm_prompt(self, text: str) -> list[Dict[str, str]]:
        tool_catalog_text = json.dumps(self.tool_catalog, ensure_ascii=False, indent=2)
        system_prompt = (
            "你是一个车载语音技能路由器。你的任务不是回答用户，而是从技能列表中选择最合适的一个技能，并提取参数。"
            "如果信息不足、用户意图不明确、或需要补充参数，就返回 need_clarification=true，并给出 clarification_question。"
            "如果是普通闲聊或不需要任何技能，selected_tool 设为 none。"
            "必须只输出 JSON，不要输出解释、Markdown 或多余文本。"
        )
        user_prompt = f"""
技能列表:
{tool_catalog_text}

请根据用户输入选择技能，并严格输出以下 JSON 结构:
{{
  "selected_tool": "skill_name 或 none",
  "arguments": {{"key": "value"}},
  "confidence": 0.0,
  "need_clarification": false,
  "clarification_question": "",
  "reason": "简短原因"
}}

约束:
1. 只能选择技能列表中的 name。
2. 车控类请求优先选择对应 vehicle_* 技能。
3. 搜索、点餐、注册声纹也必须走对应技能。
4. 不要编造参数；缺参数时请明确请求澄清。
5. confidence 取 0 到 1 之间的小数。

用户输入:
{text}
""".strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        cleaned = content.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _decision_to_intent(self, text: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        default = self._build_default_intent()
        tool_name = str(decision.get("selected_tool") or decision.get("tool") or "").strip()
        arguments = decision.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        confidence = decision.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        need_clarification = bool(decision.get("need_clarification"))
        clarification_question = str(decision.get("clarification_question") or "").strip()

        if need_clarification or confidence < self.llm_min_confidence or not tool_name or tool_name.lower() in {"none", "chat", "default"}:
            if clarification_question:
                default["Need_Clarification"] = True
                default["Clarification_Prompt"] = clarification_question
                default["Route_Source"] = "llm_clarify"
                default["Route_Confidence"] = confidence
            return default if default["Need_Clarification"] else {}

        intent = self._tool_to_legacy_intent(tool_name, arguments)
        if not intent:
            return {}

        intent["Route_Source"] = "llm"
        intent["Route_Confidence"] = confidence
        return intent

    def _tool_to_legacy_intent(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        default = self._build_default_intent()
        tool_name = tool_name.strip()

        if tool_name == "web_search":
            query = str(arguments.get("query") or arguments.get("Need_Search") or "").strip()
            if not query:
                return {}
            default["Need_Search"] = query
            return default

        if tool_name == "order_food":
            food_name = str(arguments.get("food_name") or arguments.get("food") or "").strip()
            if not food_name:
                return {}
            default["Call_elm"] = True
            default["Food_candidate"] = food_name
            return default

        if tool_name == "register_voice":
            user_name = str(arguments.get("user_name") or arguments.get("name") or "Unknown_User").strip() or "Unknown_User"
            default["Register_Action"] = user_name
            return default

        if tool_name == "vehicle_climate":
            default["Climate_Action"] = {
                "op": str(arguments.get("op") or "status"),
                "target_temp": arguments.get("target_temp"),
                "delta": arguments.get("delta"),
                "fan_speed": arguments.get("fan_speed"),
                "mode": arguments.get("mode"),
            }
            return default

        if tool_name == "vehicle_window":
            default["Window_Action"] = {
                "op": str(arguments.get("op") or "status"),
                "position": str(arguments.get("position") or "all"),
                "percent": arguments.get("percent"),
            }
            return default

        if tool_name == "vehicle_seat":
            op = str(arguments.get("op") or "status")
            default["Seat_Action"] = {
                "op": op,
                "position": str(arguments.get("position") or "driver"),
                "level": arguments.get("level"),
                "direction": arguments.get("direction"),
            }
            return default

        if tool_name == "vehicle_navigation":
            destination = str(arguments.get("destination") or "").strip()
            if not destination:
                return {}
            default["Navigation_Action"] = {
                "destination": destination,
                "waypoint": str(arguments.get("waypoint") or ""),
                "mode": str(arguments.get("mode") or "drive"),
            }
            return default

        if tool_name == "vehicle_media":
            default["Media_Action"] = {
                "op": str(arguments.get("op") or "play"),
                "source": arguments.get("source"),
                "track": arguments.get("track"),
                "volume": arguments.get("volume"),
            }
            return default

        if tool_name == "vehicle_status":
            default["Vehicle_Status_Action"] = {"op": "status"}
            return default

        return {}

    def _extract_vehicle_intent(self, text: str) -> Dict[str, Any]:
        text = text or ""
        compact = text.replace(" ", "")

        climate = self._extract_climate_action(compact)
        if climate:
            return climate

        window = self._extract_window_action(compact)
        if window:
            return window

        seat = self._extract_seat_action(compact)
        if seat:
            return seat

        navigation = self._extract_navigation_action(compact)
        if navigation:
            return navigation

        media = self._extract_media_action(compact)
        if media:
            return media

        status = self._extract_vehicle_status_action(compact)
        if status:
            return status

        return {}

    def _extract_climate_action(self, text: str) -> Dict[str, Any]:
        if not any(keyword in text for keyword in ("空调", "车内温度", "风量", "冷一点", "热一点", "制冷", "制热", "除雾")):
            return {}

        if "温度" in text and not any(keyword in text for keyword in ("空调", "车内", "车里", "调高", "调低", "设置", "设为")):
            return {}

        target_temp = None
        temp_match = re.search(r"(\d{1,2})\s*度", text)
        if temp_match:
            target_temp = int(temp_match.group(1))

        op = "status"
        if any(keyword in text for keyword in ("调高", "升高", "加一", "暖一点", "热一点", "提高")):
            op = "temp_up"
        elif any(keyword in text for keyword in ("调低", "降低", "小一点", "冷一点", "降低")):
            op = "temp_down"
        elif target_temp is not None:
            op = "set_temp"

        fan_speed = None
        fan_match = re.search(r"风量(\d+)", text)
        if fan_match:
            fan_speed = int(fan_match.group(1))

        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {
                "op": op,
                "target_temp": target_temp,
                "delta": 1 if op == "temp_up" else -1 if op == "temp_down" else None,
                "fan_speed": fan_speed,
                "mode": "auto" if "自动" in text else None,
            },
            "Window_Action": {},
            "Seat_Action": {},
            "Navigation_Action": {},
            "Media_Action": {},
            "Vehicle_Status_Action": {},
        }

    def _extract_window_action(self, text: str) -> Dict[str, Any]:
        if not any(keyword in text for keyword in ("车窗", "窗", "天窗")):
            return {}

        if any(keyword in text for keyword in ("打开", "升起", "上升", "开窗")):
            op = "open"
            percent = 100
        elif any(keyword in text for keyword in ("关闭", "关上", "落下", "升窗")):
            op = "close"
            percent = 0
        else:
            op = "status"
            percent = None

        position = "sunroof" if "天窗" in text else "all"
        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {},
            "Window_Action": {"op": op, "position": position, "percent": percent},
            "Seat_Action": {},
            "Navigation_Action": {},
            "Media_Action": {},
            "Vehicle_Status_Action": {},
        }

    def _extract_seat_action(self, text: str) -> Dict[str, Any]:
        if not any(keyword in text for keyword in ("座椅", "按摩", "加热", "通风", "靠背")):
            return {}

        if any(keyword in text for keyword in ("加热", "暖座")):
            op = "heat_on"
        elif any(keyword in text for keyword in ("通风", "降温")):
            op = "cool_on"
        elif any(keyword in text for keyword in ("按摩",)):
            op = "massage_on"
        elif any(keyword in text for keyword in ("前移", "往前")):
            op = "forward"
        elif any(keyword in text for keyword in ("后移", "往后")):
            op = "backward"
        else:
            op = "status"

        position = "passenger" if "副驾" in text else "driver"
        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {},
            "Window_Action": {},
            "Seat_Action": {"op": op, "position": position, "level": 1, "direction": op if op in ("forward", "backward") else None},
            "Navigation_Action": {},
            "Media_Action": {},
            "Vehicle_Status_Action": {},
        }

    def _extract_navigation_action(self, text: str) -> Dict[str, Any]:
        if not any(keyword in text for keyword in ("导航", "带我", "前往", "回家", "充电站", "去公司", "去学校", "去机场", "去医院", "去商场", "开到", "开去", "去往")):
            if not re.search(r"去[^，。！？?]{1,12}(家|公司|学校|机场|医院|商场|车站|充电站)", text):
                return {}

        destination = ""
        for keyword in ("回家", "去公司", "去学校", "充电站"):
            if keyword in text:
                destination = keyword.replace("去", "")
                break

        if not destination:
            destination_match = re.search(r"(导航到|前往|去往|带我去|开到|开去|去|到)([^，。！？?]+)", text)
            if destination_match:
                destination = destination_match.group(2)

        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {},
            "Window_Action": {},
            "Seat_Action": {},
            "Navigation_Action": {"destination": destination or "目的地", "waypoint": "", "mode": "drive"},
            "Media_Action": {},
            "Vehicle_Status_Action": {},
        }

    def _extract_media_action(self, text: str) -> Dict[str, Any]:
        if not any(keyword in text for keyword in ("音乐", "播放", "暂停", "下一首", "上一首", "音量", "切歌", "听歌")):
            return {}

        if any(keyword in text for keyword in ("暂停", "停止")):
            op = "pause"
        elif any(keyword in text for keyword in ("下一首", "下一曲")):
            op = "next"
        elif any(keyword in text for keyword in ("上一首", "上一曲")):
            op = "prev"
        else:
            op = "play"

        volume = None
        volume_match = re.search(r"音量(\d{1,2})", text)
        if volume_match:
            volume = int(volume_match.group(1))

        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {},
            "Window_Action": {},
            "Seat_Action": {},
            "Navigation_Action": {},
            "Media_Action": {"op": op, "source": "local", "track": "", "volume": volume},
            "Vehicle_Status_Action": {},
        }

    def _extract_vehicle_status_action(self, text: str) -> Dict[str, Any]:
        if not any(keyword in text for keyword in ("车况", "胎压", "续航", "油量", "电量", "保养", "车辆状态")):
            return {}

        return {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": "",
            "Climate_Action": {},
            "Window_Action": {},
            "Seat_Action": {},
            "Navigation_Action": {},
            "Media_Action": {},
            "Vehicle_Status_Action": {"op": "status"},
        }


def run_router_smoke_tests() -> None:
    import asyncio
    from types import SimpleNamespace

    class _FakeCompletions:
        def __init__(self, content: str):
            self.content = content

        async def create(self, *args, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
            )

    class _FakeClient:
        def __init__(self, content: str):
            self.chat = SimpleNamespace(completions=_FakeCompletions(content))

    class _FallbackRouter:
        def route(self, text: str):
            default = {
                "Call_elm": False,
                "Food_candidate": "",
                "Need_Search": "",
                "Register_Action": "",
                "Climate_Action": {},
                "Window_Action": {},
                "Seat_Action": {},
                "Navigation_Action": {},
                "Media_Action": {},
                "Vehicle_Status_Action": {},
            }
            if any(keyword in text for keyword in ("天气", "新闻", "百科")):
                default["Need_Search"] = text.strip()
            elif any(keyword in text for keyword in ("注册", "声纹", "我是")):
                default["Register_Action"] = "小明"
            elif any(keyword in text for keyword in ("想吃", "外卖", "来一份")):
                default["Call_elm"] = True
                default["Food_candidate"] = "汉堡"
            return default

    class _DummySkillRegistry:
        def get_all_tools(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "vehicle_climate",
                        "description": "调整空调温度、风量和模式。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string"},
                                "target_temp": {"type": "integer"},
                                "delta": {"type": "integer"},
                                "fan_speed": {"type": "integer"},
                                "mode": {"type": "string"},
                            },
                            "required": [],
                            "x-llm-hints": {
                                "required": [],
                                "optional": ["op", "target_temp", "delta", "fan_speed", "mode"],
                                "examples": [],
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vehicle_navigation",
                        "description": "发起导航到目的地。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "destination": {"type": "string"},
                                "waypoint": {"type": "string"},
                                "mode": {"type": "string"},
                            },
                            "required": ["destination"],
                            "x-llm-hints": {
                                "required": ["destination"],
                                "optional": ["waypoint", "mode"],
                                "examples": [],
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "register_voice",
                        "description": "注册声纹。",
                        "parameters": {
                            "type": "object",
                            "properties": {"user_name": {"type": "string"}},
                            "required": ["user_name"],
                            "x-llm-hints": {
                                "required": ["user_name"],
                                "optional": [],
                                "examples": [],
                            },
                        },
                    },
                },
            ]

    def _assert_case(name: str, condition: bool, payload: Dict[str, Any]) -> None:
        if not condition:
            raise AssertionError(f"{name} failed: {payload}")
        print(f"[PASS] {name}")

    async def _run() -> None:
        cases = [
            {
                "name": "llm_climate_route",
                "text": "有点冷，帮我调到24度",
                "router": IntentRouterService(
                    _FallbackRouter(),
                    llm_client=_FakeClient('{"selected_tool":"vehicle_climate","arguments":{"op":"set_temp","target_temp":24},"confidence":0.98,"need_clarification":false,"clarification_question":"","reason":""}'),
                    llm_model="dummy",
                    skill_registry=_DummySkillRegistry(),
                    llm_enabled=True,
                ),
                "check": lambda result: result.get("Route_Source") == "llm" and result.get("Climate_Action", {}).get("target_temp") == 24,
            },
            {
                "name": "llm_clarification_route",
                "text": "帮我导航",
                "router": IntentRouterService(
                    _FallbackRouter(),
                    llm_client=_FakeClient('{"selected_tool":"vehicle_navigation","arguments":{},"confidence":0.90,"need_clarification":true,"clarification_question":"你想去哪里？","reason":"缺少目的地"}'),
                    llm_model="dummy",
                    skill_registry=_DummySkillRegistry(),
                    llm_enabled=True,
                ),
                "check": lambda result: result.get("Need_Clarification") is True and result.get("Clarification_Prompt") == "你想去哪里？",
            },
            {
                "name": "heuristic_climate_route",
                "text": "把空调调高一点",
                "router": IntentRouterService(
                    _FallbackRouter(),
                    llm_client=None,
                    llm_model="",
                    skill_registry=None,
                    llm_enabled=False,
                ),
                "check": lambda result: result.get("Route_Source") == "heuristic" and result.get("Climate_Action", {}).get("op") == "temp_up",
            },
            {
                "name": "router_fallback_register",
                "text": "我是张三，帮我注册声纹",
                "router": IntentRouterService(
                    _FallbackRouter(),
                    llm_client=None,
                    llm_model="",
                    skill_registry=None,
                    llm_enabled=False,
                ),
                "check": lambda result: result.get("Route_Source") == "bert" and result.get("Register_Action") == "小明",
            },
        ]

        for case in cases:
            result = await case["router"].route_async(case["text"])
            _assert_case(case["name"], case["check"](result), result)

    asyncio.run(_run())


if __name__ == "__main__":
    run_router_smoke_tests()
