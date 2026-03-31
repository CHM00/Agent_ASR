import os
import json
import asyncio
import re
import threading
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from openai import AsyncOpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from Milvus import MilvusClass
from tavily import TavilyClient
from Local_Model import Load_Model
from Knowledge_Grpah import KnowledgeGraph
from typing import AsyncGenerator, List, Dict, Tuple

# 加载环境变量(.env文件)
load_dotenv()


class SmartAgentBrain:
    def __init__(self, LOCAL_LLM=False):
        # ================= 配置区域 =================
        self.ARK_API_KEY = os.environ.get("ARK_API_KEY")
        self.ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.LLM_MODEL = "deepseek-ai/DeepSeek-V3"
        self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

        self.MILVUS_URI = os.environ.get("URL")
        self.MILVUS_TOKEN = os.environ.get("Token")

        self.search_web_key = os.environ.get("trivily_key")

        # 初始化 LLM 客户端
        self.aclient = AsyncOpenAI(
            api_key=self.ARK_API_KEY,
            base_url=self.ARK_BASE_URL,
        )

        # 检索工具
        self.client = TavilyClient(self.search_web_key)

        # === 上下文与压缩配置 ===
        self.history = []
        self.max_history_len = 20  # 放宽历史记录上限，交由动态预算管理器处理
        self.running_summary = ""  # 滚动摘要，存储被折叠的旧历史
        self.max_context_tokens = 1600  # 最大 Token 预算

        # 初始化类
        self.milvus = MilvusClass()
        self.milvus.connect_milvus()
        self.memory_collection = self.milvus.memory_collection
        self.collection = self.milvus.food_collection

        # 初始化图谱
        self.kg = KnowledgeGraph()
        self.kg.connect()

        # 初始化本地类 (仅保留 ASR/CAM 等非 LLM 本地模型)
        self.local_model = Load_Model()
        self.LOCAL_LLM = LOCAL_LLM  # 强制使用 API 模型

        from intent_router_bert import BertIntentRouter
        self.intent_router = BertIntentRouter(model_dir="./BERT-Finetuing/final_intent_model")


    def _estimate_tokens(self, text: str) -> int:
        """中文场景粗估 token 数量"""
        return max(1, int(len(text) / 1.8)) if text else 0

    def _estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        joined = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        return self._estimate_tokens(joined)

    async def _compress_text_api(self, text: str, max_chars: int = 450) -> str:
        """调用 DeepSeek-V3 压缩长文本（如检索结果）"""
        if not text or len(text) <= max_chars:
            return text

        prompt = [
            {"role": "system",
             "content": "你是上下文压缩器。请在不丢失关键信息的前提下压缩文本。必须保留：人物/地点/时间/数值/用户明确要求/待办事项。输出中文，禁止编造。"},
            {"role": "user", "content": text}
        ]
        try:
            res = await self.aclient.chat.completions.create(
                model=self.LLM_MODEL, messages=prompt, temperature=0.1
            )
            summary = res.choices[0].message.content.strip()
            return summary[:max_chars]
        except Exception as e:
            print(f"文本压缩失败: {e}")
            return text[:max_chars]

    async def _compress_messages_api(self, messages: List[Dict[str, str]], max_chars: int = 400) -> str:
        """调用 DeepSeek-V3 压缩旧对话，提取核心摘要"""
        if not messages:
            return ""

        raw = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        prompt = [
            {"role": "system",
             "content": "你是对话记忆压缩器。请提炼这段对话的核心脉络、稳定事实和未完成问题。要求极致精简，不超过200字，中文输出。"},
            {"role": "user", "content": raw}
        ]
        try:
            res = await self.aclient.chat.completions.create(
                model=self.LLM_MODEL, messages=prompt, temperature=0.1
            )
            summary = res.choices[0].message.content.strip()
            return summary[:max_chars]
        except Exception as e:
            print(f"对话压缩失败: {e}")
            return ""

    async def _build_context_api(self, system_prompt: str, user_input: str, memory_str: str = "",
                                 search_ctx: str = "") -> Tuple[List[Dict[str, str]], str]:
        """
        分级预算组装上下文，返回 (组装好的messages, 更新后的滚动摘要)
        """
        messages = [{"role": "system", "content": system_prompt}]
        new_running_summary = self.running_summary

        # 拼接基础系统提示
        if memory_str:
            messages[0]["content"] += f"\n{memory_str}\n请在聊天中自然运用这些信息。"
        if self.running_summary:
            messages.append({"role": "system", "content": f"【历史摘要】:\n{self.running_summary}"})
        if search_ctx:
            messages.append({"role": "system", "content": f"【检索上下文】:\n{search_ctx}"})

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})

        # 第 0 级：未超标，直接返回, 每一次都评估当前上下文的token数量，如果未超标则直接使用，不进行压缩
        total_tokens = self._estimate_messages_tokens(messages)
        if total_tokens <= self.max_context_tokens:
            return messages, new_running_summary

        print(f"[Context] Token 超标 ({total_tokens} > {self.max_context_tokens})，触发第一级压缩 (外部知识)...")

        # 第 1 级：压缩检索上下文, 因为它通常最冗长且信息密度较低，优先压缩 (说明: 如果网页检索结果非常简洁，压缩后反而更冗长，可以考虑增加一个阈值判断是否需要压缩)
        if search_ctx:
            compressed_search = await self._compress_text_api(search_ctx, max_chars=300)
            # 替换旧的检索上下文
            messages = [m for m in messages if not m["content"].startswith("【检索上下文】:")]
            # 插入到 user 问题之前，或者 system prompt 之后
            messages.insert(1, {"role": "system", "content": f"【检索上下文(压缩)】:\n{compressed_search}"})

        total_tokens = self._estimate_messages_tokens(messages)
        if total_tokens <= self.max_context_tokens:
            return messages, new_running_summary

        print(f"[Context] Token 依然超标 ({total_tokens})，触发第二级压缩 (折叠历史对话)...")

        # 第 2 级：压缩历史记录 (保留最近2轮，也就是 4 条 message)， 保留最新的4条message, 将更早的历史进行压缩摘要，并合并到滚动摘要中
        keep_recent = self.history[-4:] if len(self.history) > 4 else self.history
        old_recent = self.history[:-4] if len(self.history) > 4 else []

        if old_recent:
            old_summary = await self._compress_messages_api(old_recent)
            # 合并新旧摘要
            new_running_summary = (self.running_summary + "\n" + old_summary).strip()[:800]
            # 更新历史记录，剔除被折叠的部分
            self.history = keep_recent

        # 重新组装终极报文
        final_msgs = [{"role": "system", "content": system_prompt}]
        if memory_str:
            final_msgs[0]["content"] += f"\n{memory_str}\n请在聊天中自然运用这些信息。"
        if new_running_summary:
            final_msgs.append({"role": "system", "content": f"【历史摘要】:\n{new_running_summary}"})
        if search_ctx:
            final_msgs.append({"role": "system", "content": f"【检索上下文(压缩)】:\n{compressed_search}"})

        final_msgs.extend(keep_recent)
        final_msgs.append({"role": "user", "content": user_input})

        return final_msgs, new_running_summary

    async def stream_chat_llm(self, messages, temperature=0.7) -> AsyncGenerator[str, None]:
        """封装 LLM 流式调用接口"""
        if self.LOCAL_LLM:
            # 本地模型目前在 Load_Model 中是同步推理，这里模拟流式输出
            # 如果本地模型支持流式，请在此处替换为对应的流式 generate 逻辑
            full_reply = self.local_model.llm_chat(messages)
            yield full_reply
        else:
            try:
                response = await self.aclient.chat.completions.create(
                    model=self.LLM_MODEL,
                    messages=messages,
                    stream=True,
                    temperature=temperature
                )
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            except Exception as e:
                yield f"连接模型出错：{e}"

    async def process_user_query(self, user_text, user_id) -> AsyncGenerator[str, None]:
        """核心意图处理函数：带有上下文动态压缩"""
        print(f"处理请求 [流式]，用户: {user_id}")

        # 1. 召回记忆是在意图识别之前
        related_memories = self.recall_memories(user_text, user_id)
        memory_str = f"【关于 {user_id} 的记忆】: {';'.join(related_memories)}" if related_memories else ""

        # 2. 意图路由
        intent = await self._route_intent(user_text)
        print(f"意图分析: {intent}")

        full_response = ""
        sentence_buffer = ""
        split_punc = ["。", "！", "？", "；", "...", ".", "!", "?"]

        # --- 分支 A: 点餐业务 ---
        if intent.get("Call_elm"):
            food_name = intent.get("Food_candidate")
            matched = self.search_food_db(food_name)
            reply = f"找到啦！我们要不要来一份{matched}？" if matched else f"抱歉，菜单里没有{food_name}。"
            yield reply
            full_response = reply

        # --- 分支 B: 联网搜索 ---
        elif intent.get("Need_Search"):
            search_ctx = self.search_web(intent.get("Need_Search"))
            system_msg = f"你叫小千，是一个活泼可爱的语音助手。请结合上下文极简回答用户，不超过30字。"

            # 使用上下文压缩引擎构建 Context
            msgs, self.running_summary = await self._build_context_api(
                system_prompt=system_msg,
                user_input=user_text,
                memory_str=memory_str,
                search_ctx=search_ctx
            )

            async for chunk in self.stream_chat_llm(msgs, temperature=0.1):
                sentence_buffer += chunk
                if any(p in chunk for p in split_punc):
                    yield sentence_buffer.strip()
                    full_response += sentence_buffer
                    sentence_buffer = ""

        # --- 分支 C: 注册声纹 ---
        elif intent.get("Register_Action"):
            yield f"ACTION_REGISTER:{intent.get('Register_Action')}"
            return

        # --- 分支 D: 闲聊/通用对话 ---
        else:
            system_msg = f"你叫小千，是一个活泼可爱的语音助手，现在的对话者是 {user_id}, 回答请简短口语化。"

            # 使用上下文压缩引擎构建 Context
            msgs, self.running_summary = await self._build_context_api(
                system_prompt=system_msg,
                user_input=user_text,
                memory_str=memory_str
            )

            async for chunk in self.stream_chat_llm(msgs):
                sentence_buffer += chunk
                if any(p in chunk for p in split_punc):
                    yield sentence_buffer.strip()
                    full_response += sentence_buffer
                    sentence_buffer = ""

        # 产出剩余的 buffer
        if sentence_buffer.strip():
            yield sentence_buffer.strip()
            full_response += sentence_buffer

        # 更新历史与后台记忆任务
        self._post_process(user_text, full_response, user_id)

    async def _route_intent(self, user_text: str) -> dict:
        """用 BERT 本地路由替代 LLM Prompt 路由"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.intent_router.route, user_text)
        return result

    def _post_process(self, user_text, full_response, user_id):
        """更新历史与启动异步记忆提取任务"""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})
        # 截断策略放宽，依赖 build_context_api 动态压缩
        if len(self.history) > self.max_history_len:
            self.history = self.history[-self.max_history_len:]

        # 在守护线程中运行记忆存储
        def run_mem():
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.extract_and_save_memory(user_text, user_id))
            except Exception as e:
                print(f"后台记忆存储失败: {e}")

        threading.Thread(target=run_mem, daemon=True).start()


    async def extract_and_save_memory(self, user_text, user_id):
        print(f"正在分析用户记忆: {user_text}")
        prompt = f"""
        ### 任务
        从用户的话中提取**长期用户画像**，并转化为结构化的 JSON 数据以便存入知识图谱。

        ### 提取规则
        1. 识别用户 (User) 与 实体 (Entity) 之间的关系。
        2. 仅提取以下关系类型：
           - LIKES/DISLIKES: 喜好
           - ALLERGY: 过敏
           - HABIT: 习惯
           - IS_A: 身份/职业 (如: 我是学生)
           - LIVES_IN: 居住地 (如: 我住在北京, 我定居上海)  <-- 新增强调
           - STATUS: 状态 (如: 我单身, 我刚搬家)
        3. 如果没有长期有效的信息，输出 "NONE"。

        ### 输出格式 (JSON List)
        [
            {{"relation": "RELATION_TYPE", "target": "Entity_Name", "type": "Entity_Type"}}
        ]

        ### 示例
        输入: "我不吃香菜，我对花生过敏"
        输出: [
            {{"relation": "DISLIKES", "target": "香菜", "type": "Food"}},
            {{"relation": "ALLERGY", "target": "花生", "type": "Ingredient"}}
        ]

        输入: "我是一个程序员"
        输出: [{{"relation": "IS_A", "target": "程序员", "type": "Job"}}]

        输入: "今天天气不错"
        输出: NONE

        ### 用户输入
        "{user_text}"

        ### 你的输出 (仅JSON)
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            res = await self.aclient.chat.completions.create(
                model=self.LLM_MODEL, messages=messages, temperature=0.0
            )
            content = res.choices[0].message.content.strip()

            if "NONE" not in content and "{" in content:
                content = content.replace("```json", "").replace("```", "")
                triplets = json.loads(content)

                for t in triplets:
                    new_relation = t.get("relation")
                    target_name = t.get("target")
                    t_type = t.get("type", "Entity")

                    fact_text = f"用户 {new_relation} {target_name}"
                    similar_memories = self.milvus.search_memory(fact_text, user_id, top_k=10)

                    if t_type == "Food" and new_relation == "LIKES":
                        allergy_res = self.milvus.search_memory(f"用户 ALLERGY {target_name}", user_id, top_k=3)
                        existing_ids = set(m['id'] for m in similar_memories)
                        for am in allergy_res:
                            if am['id'] not in existing_ids:
                                similar_memories.append(am)

                    decision = await self.detect_conflict_with_llm(fact_text, similar_memories, user_input=user_text)

                    if decision['action'] == "IGNORE":
                        continue

                    ids_to_delete = decision.get("ids", [])
                    if decision['action'] == "DELETE" and ids_to_delete:
                        self.milvus.delete_memory_by_ids(ids_to_delete, user_id)
                        for mid in ids_to_delete:
                            self.kg.delete_relation_by_mid(mid)

                    milvus_id = self.milvus.insert_memory(fact_text, user_id)
                    if milvus_id:
                        self.kg.upsert_relation(user_id, new_relation, target_name, t_type, milvus_id)
        except Exception as e:
            print(f" 记忆提取或图谱写入失败: {e}")

    async def detect_conflict_with_llm(self, new_memory_text, existing_memories_from_milvus, user_input=""):
        if not existing_memories_from_milvus: return {"action": "NONE"}

        context_str = "\n".join([
            f"ID:{m['id']} | 内容: {m['text']} (相似度:{m['score']:.2f})"
            for m in existing_memories_from_milvus
        ])

        prompt = f"""
                你是一个记忆一致性管理员。请根据【原始对话】和【新提取信息】，判断与【现有记忆】的冲突。

                【原始对话语境】(最高优先级):
                "{user_input}"
                (注意：如果用户在对话中明确表示了"不再"、"搬家"、"换工作"、"改做"等变更意图，请坚决删除旧状态。)

                【现有记忆】:
                {context_str}

                【新提取信息】:
                {new_memory_text}

                请严格根据以下规则裁决：
                1. **显式终止 (DELETE)**: 
                   - 原始对话中出现 "不让做了"、"戒了"、"改喝..." 等，必须删除旧习惯。
                   - 例如：原话"医生不让跑步了，改游泳"，旧记忆"晨跑" -> DELETE。
                2. **状态/身份变更 (DELETE)**: 
                   - 原始对话中出现 "考上公务员"、"回老家"、"搬家"，意味着旧的 "工作压力大"、"住在北京" 等状态已失效 -> DELETE。
                3. **属性冲突 (DELETE)**: 
                   - "喜欢辣" vs "一点辣都不能吃" -> DELETE。
                4. **冗余 (IGNORE)**: 内容完全一致。
                5. **共存 (NONE)**: 无逻辑冲突。

                输出格式 (JSON):
                - 需删除: {{"action": "DELETE", "ids": [123]}}
                - 冗余: {{"action": "IGNORE"}}
                - 无操作: {{"action": "NONE"}}
                """
        try:
            messages = [{"role": "user", "content": prompt}]
            res = await self.aclient.chat.completions.create(
                model=self.LLM_MODEL, messages=messages, temperature=0.0
            )
            content = res.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"action": "NONE"}
        except Exception as e:
            print(f" 裁决失败: {e}")
            return {"action": "NONE"}

    def recall_memories(self, query, user_id):
        final_results = []
        vec_res = self.milvus.search_memory(query, user_id, top_k=3)
        for r in vec_res:
            final_results.append(f"[语义] {r['text']}")
        graph_res = self.kg.search_user_graph(user_id)
        for r in graph_res:
            final_results.append(f"[图谱] {r}")
        return list(set(final_results))

    def search_food_db(self, query_text):
        return self.kg.search_food(query_text)

    def search_web(self, query, max_results=2):
        print(f"[Brain] 正在联网搜索: {query} ...")
        for attempt in range(3):
            try:
                response = self.client.search(query=query, search_depth="basic")
                results = response["results"][:max_results]
                if not results: return ""
                compact = []
                for r in results:
                    title = (r.get("title") or "")[:40]
                    content = (r.get("content") or "").replace("\n", " ")
                    content = " ".join(content.split())[:160]
                    compact.append(f"标题:{title} 摘要:{content}")
                return "\n".join(compact)
            except Exception as e:
                print(f"[Brain] 搜索尝试 {attempt + 1} 失败: {e}")
                if attempt < 2:
                    import time
                    time.sleep(1)
        return ""


if __name__ == "__main__":
    brain = SmartAgentBrain()
    asyncio.run(brain.extract_and_save_memory("最近膝盖受伤了，医生不让跑步了，以后早上改游泳了。", "主人"))