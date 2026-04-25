import json
from tavily import TavilyClient
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from mcp_gateway import MCPGateway


@dataclass
class SkillResult:
    status: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseSkill:
    """技能基类，所有具体的技能都继承此类"""
    name = ""
    description = ""
    parameters = {}
    required_parameters = []
    optional_parameters = []
    examples = []
    risk_level = "low"
    timeout_ms = 3000
    requires_auth = False
    idempotent = True

    async def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def get_tool_schema(self) -> dict:
        """返回供大模型识别的 Tool Schema"""
        required_parameters = list(self.required_parameters)
        optional_parameters = list(self.optional_parameters) if self.optional_parameters else [
            key for key in self.parameters.keys() if key not in required_parameters
        ]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": required_parameters,
                    "additionalProperties": False,
                    "x-llm-hints": {
                        "required": required_parameters,
                        "optional": optional_parameters,
                        "examples": self.examples,
                    },
                }
            }
        }


class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "当用户询问实时信息、天气、新闻、百科等需要联网查询的内容时调用此技能。"
    required_parameters = ["query"]
    optional_parameters = []
    examples = [
        {"input": "明天北京天气怎么样", "arguments": {"query": "明天北京天气怎么样"}},
        {"input": "查一下 DeepSeek 最新模型", "arguments": {"query": "DeepSeek 最新模型"}},
    ]
    parameters = {
        "query": {
            "type": "string",
            "description": "需要搜索引擎查询的具体关键词，例如'北京今天天气'或'DeepSeek最新模型'"
        }
    }

    def __init__(self):
        api_key = os.environ.get("trivily_key") or os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key) if api_key else None

    async def execute(self, query: str, **kwargs) -> str:
        print(f"[Skill] 执行联网搜索: {query}")
        if not self.client:
            return "当前未配置联网搜索密钥，无法执行搜索。"
        try:
            # 使用现有的 tavily 逻辑
            response = self.client.search(query=query, search_depth="basic")
            results = response.get("results", [])[:2]
            if not results:
                return "未检索到相关信息。"

            compact = []
            for r in results:
                title = r.get("title", "")[:40]
                content = r.get("content", "").replace("\n", " ")[:160]
                compact.append(f"【{title}】{content}")
            return "\n".join(compact)
        except Exception as e:
            return f"搜索失败: {str(e)}"


class FoodDeliverySkill(BaseSkill):
    name = "order_food"
    description = "当用户表达想吃什么、点外卖、饿了等意图时调用此技能。"
    required_parameters = ["food_name"]
    optional_parameters = []
    examples = [
        {"input": "我想吃汉堡", "arguments": {"food_name": "汉堡"}},
        {"input": "来一份宫保鸡丁", "arguments": {"food_name": "宫保鸡丁"}},
    ]
    parameters = {
        "food_name": {
            "type": "string",
            "description": "用户想吃的具体食物名称，例如'汉堡'、'宫保鸡丁'"
        }
    }

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph  # 依赖注入图谱实例

    async def execute(self, food_name: str, **kwargs) -> str:
        print(f"[Skill] 执行点餐逻辑: {food_name}")
        # 模拟你的图谱查询逻辑
        matched = self.kg.search_food(food_name) if hasattr(self.kg, 'search_food') else food_name
        if matched:
            return f"系统已为您在菜单中找到【{matched}】，即将为您下单。"
        return f"抱歉，当前的食材库中没有找到【{food_name}】。"


class RegisterVoiceSkill(BaseSkill):
    name = "register_voice"
    description = "当用户要求注册声纹、记录身份、或说'我是谁谁谁'时调用此技能。"
    required_parameters = ["user_name"]
    optional_parameters = []
    examples = [
        {"input": "我是张三", "arguments": {"user_name": "张三"}},
        {"input": "帮我注册声纹，叫我小明", "arguments": {"user_name": "小明"}},
    ]
    parameters = {
        "user_name": {
            "type": "string",
            "description": "用户声明的名字，如果没有提到具体名字，提取为'Unknown_User'"
        }
    }

    async def execute(self, user_name: str, **kwargs) -> str:
        # 这个技能比较特殊，它主要是返回一个控制指令给客户端
        print(f"[Skill] 触发声纹注册: {user_name}")
        return f"ACTION_REGISTER:{user_name}"


class VehicleBaseSkill(BaseSkill):
    """车载技能基类，统一通过 MCP 网关访问车控总线。"""

    tool_name = ""

    def __init__(self, gateway=None):
        self.gateway = gateway or MCPGateway()

    def _invoke(self, payload: Dict[str, Any]) -> str:
        result = self.gateway.invoke(self.tool_name, payload)
        if result.success:
            return result.message
        return f"车控执行失败: {result.message}"


class ClimateControlSkill(VehicleBaseSkill):
    name = "vehicle_climate"
    tool_name = "vehicle_climate"
    description = "调整空调温度、风量和模式，例如升高温度、降低温度、设置风量。"
    required_parameters = []
    optional_parameters = ["op", "target_temp", "delta", "fan_speed", "mode"]
    examples = [
        {"input": "有点冷，调高一点温度", "arguments": {"op": "temp_up", "delta": 1}},
        {"input": "把空调调到24度", "arguments": {"op": "set_temp", "target_temp": 24}},
        {"input": "风量调到3档", "arguments": {"op": "set_fan", "fan_speed": 3}},
        {"input": "切到自动空调", "arguments": {"op": "set_mode", "mode": "auto"}},
    ]
    parameters = {
        "op": {"type": "string", "description": "操作类型，例如 temp_up、temp_down、status"},
        "target_temp": {"type": "integer", "description": "目标温度，例如 24"},
        "delta": {"type": "integer", "description": "相对调节幅度，例如 +1 或 -1"},
        "fan_speed": {"type": "integer", "description": "风量档位，1 到 7"},
        "mode": {"type": "string", "description": "模式，例如 auto、cool、heat"},
    }

    async def execute(self, **kwargs) -> str:
        return self._invoke(kwargs)


class WindowControlSkill(VehicleBaseSkill):
    name = "vehicle_window"
    tool_name = "vehicle_window"
    description = "控制车窗或天窗的开合、升降和百分比位置。"
    required_parameters = []
    optional_parameters = ["op", "position", "percent"]
    examples = [
        {"input": "打开车窗", "arguments": {"op": "open", "position": "all", "percent": 100}},
        {"input": "关闭天窗", "arguments": {"op": "close", "position": "sunroof", "percent": 0}},
        {"input": "把左前窗调到一半", "arguments": {"op": "set_position", "position": "front_left", "percent": 50}},
    ]
    parameters = {
        "op": {"type": "string", "description": "操作类型，例如 open、close、up、down"},
        "position": {"type": "string", "description": "位置，例如 all、front_left、sunroof"},
        "percent": {"type": "integer", "description": "开合百分比，0 到 100"},
    }

    async def execute(self, **kwargs) -> str:
        return self._invoke(kwargs)


class SeatControlSkill(VehicleBaseSkill):
    name = "vehicle_seat"
    tool_name = "vehicle_seat"
    description = "控制座椅加热、通风、按摩和位置调整。"
    required_parameters = []
    optional_parameters = ["op", "position", "level", "direction"]
    examples = [
        {"input": "打开主驾座椅加热", "arguments": {"op": "heat_on", "position": "driver", "level": 1}},
        {"input": "副驾座椅通风开到2档", "arguments": {"op": "cool_on", "position": "passenger", "level": 2}},
        {"input": "主驾座椅往前调一点", "arguments": {"op": "forward", "position": "driver", "direction": "forward"}},
        {"input": "打开按摩", "arguments": {"op": "massage_on", "position": "driver", "level": 1}},
    ]
    parameters = {
        "op": {"type": "string", "description": "操作类型，例如 heat_on、cool_on、massage_on、forward"},
        "position": {"type": "string", "description": "座椅位置，例如 driver、passenger"},
        "level": {"type": "integer", "description": "档位，例如 1 到 3"},
        "direction": {"type": "string", "description": "方向，例如 forward、backward"},
    }

    async def execute(self, **kwargs) -> str:
        return self._invoke(kwargs)


class NavigationSkill(VehicleBaseSkill):
    name = "vehicle_navigation"
    tool_name = "vehicle_navigation"
    description = "发起导航到目的地，可选途经点。"
    required_parameters = ["destination"]
    optional_parameters = ["waypoint", "mode"]
    examples = [
        {"input": "导航到上海虹桥火车站", "arguments": {"destination": "上海虹桥火车站", "mode": "drive"}},
        {"input": "带我去公司", "arguments": {"destination": "公司", "mode": "drive"}},
        {"input": "导航到机场，途经充电站", "arguments": {"destination": "机场", "waypoint": "充电站", "mode": "drive"}},
    ]
    parameters = {
        "destination": {"type": "string", "description": "目的地，例如 家、公司、充电站"},
        "waypoint": {"type": "string", "description": "途经点"},
        "mode": {"type": "string", "description": "导航模式，例如 drive、walk"},
    }

    async def execute(self, **kwargs) -> str:
        return self._invoke(kwargs)


class MediaControlSkill(VehicleBaseSkill):
    name = "vehicle_media"
    tool_name = "vehicle_media"
    description = "控制车机媒体播放、暂停、切歌和音量。"
    required_parameters = []
    optional_parameters = ["op", "source", "track", "volume"]
    examples = [
        {"input": "播放音乐", "arguments": {"op": "play", "source": "local"}},
        {"input": "下一首", "arguments": {"op": "next"}},
        {"input": "音量调到16", "arguments": {"op": "set_volume", "volume": 16}},
        {"input": "切换到蓝牙", "arguments": {"op": "set_source", "source": "bluetooth"}},
    ]
    parameters = {
        "op": {"type": "string", "description": "操作类型，例如 play、pause、next、prev"},
        "source": {"type": "string", "description": "媒体来源，例如 local、bluetooth、radio"},
        "track": {"type": "string", "description": "指定曲目或内容"},
        "volume": {"type": "integer", "description": "音量大小，0 到 30"},
    }

    async def execute(self, **kwargs) -> str:
        return self._invoke(kwargs)


class VehicleStatusSkill(VehicleBaseSkill):
    name = "vehicle_status"
    tool_name = "vehicle_status"
    description = "查询车辆状态，包括胎压、续航、电量、油量和保养信息。"
    required_parameters = []
    optional_parameters = []
    examples = [
        {"input": "车辆状态怎么样", "arguments": {}},
        {"input": "查一下胎压和续航", "arguments": {}},
    ]
    parameters = {}

    async def execute(self, **kwargs) -> str:
        return self._invoke({})


class SkillRegistry:
    """技能注册中心"""
    def __init__(self, kg_instance, vehicle_gateway=None):
        self.vehicle_gateway = vehicle_gateway or MCPGateway()
        self.skills = {
            "web_search": WebSearchSkill(),
            "order_food": FoodDeliverySkill(kg_instance),
            "register_voice": RegisterVoiceSkill(),
            "vehicle_climate": ClimateControlSkill(self.vehicle_gateway),
            "vehicle_window": WindowControlSkill(self.vehicle_gateway),
            "vehicle_seat": SeatControlSkill(self.vehicle_gateway),
            "vehicle_navigation": NavigationSkill(self.vehicle_gateway),
            "vehicle_media": MediaControlSkill(self.vehicle_gateway),
            "vehicle_status": VehicleStatusSkill(self.vehicle_gateway),
        }

    def get_all_tools(self):
        return [skill.get_tool_schema() for skill in self.skills.values()]

    async def execute_tool_result(self, tool_name: str, arguments: dict) -> SkillResult:
        skill = self.skills.get(tool_name)
        if not skill:
            return SkillResult(status="error", message="未知技能", error=f"skill_not_found:{tool_name}")

        try:
            result = await skill.execute(**arguments)
            return SkillResult(status="ok", message=result, data={"tool": tool_name, "arguments": arguments})
        except Exception as e:
            return SkillResult(status="error", message="技能执行失败", error=str(e), data={"tool": tool_name})

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        result = await self.execute_tool_result(tool_name, arguments)
        if result.status == "ok":
            return result.message
        return f"{result.message}: {result.error}" if result.error else result.message