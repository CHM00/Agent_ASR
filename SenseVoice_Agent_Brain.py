# filename: agent_brain.py
import os
import json
import asyncio
import requests
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from openai import AsyncOpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from Milvus import MilvusClass

# 加载环境变量(.env文件)
load_dotenv()
# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'


class SmartAgentBrain:
    def __init__(self):
        # ================= 配置区域 (请修改这里) =================
        self.ARK_API_KEY = os.environ.get("ARK_API_KEY")  # 填入你的 API Key
        self.ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.LLM_MODEL = "deepseek-ai/DeepSeek-V3"  # 你的推理模型 ID
        self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"  # 你的 Embedding 模型 ID

        self.MILVUS_URI = os.environ.get("URL")  # Milvus 地址
        self.MILVUS_TOKEN = os.environ.get("Token")  # Milvus Token
        # =======================================================

        # 初始化 LLM 客户端
        self.aclient = AsyncOpenAI(
            api_key=self.ARK_API_KEY,
            base_url=self.ARK_BASE_URL,
        )

        # 初始化搜索工具
        self.ddgs = DDGS(proxy="http://127.0.0.1:7897")


        # self.connect_milvus()

        self.history = []  # [新增] 短期历史记录窗口
        self.max_history_len = 6  # 只保留最近6轮对话

        # 初始化类
        self.milvus = MilvusClass()
        self.milvus.connect_milvus()
        self.memory_collection = self.milvus.memory_collection
        self.collection = self.milvus.food_collection


    # def connect_milvus(self):
    #     """连接 Milvus 数据库并初始化记忆集合"""
    #     try:
    #         connections.connect(alias="link", uri=self.MILVUS_URI, token=self.MILVUS_TOKEN)
    #         print("✅ [Brain] Milvus 连接成功")
    #
    #         # 1. 加载原有的食物数据库 (保持不变)
    #         self.collection = Collection(name="MilVus_test", using="link")
    #         self.collection.load()
    #
    #         # 2. [新增] 初始化用户记忆集合 'User_Memory'
    #         self.init_memory_collection()
    #
    #     except Exception as e:
    #         print(f"⚠️ [Brain] Milvus 连接失败或集合初始化错: {e}")
    #         self.collection = None

    # def init_memory_collection(self):
    #     """[新增] 创建或加载用户记忆集合"""
    #     mem_name = "User_Memory"
    #     if utility.has_collection(mem_name, using="link"):
    #         self.memory_collection = Collection(mem_name, using="link")
    #         self.memory_collection.load()
    #         print(f"🧠 [Memory] 加载长期记忆库: {mem_name}")
    #     else:
    #         # 定义 Schema
    #         fields = [
    #             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    #             FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2560),  # 维度需与 Embedding 模型一致
    #             FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),  # 存储记忆文本
    #             FieldSchema(name="timestamp", dtype=DataType.INT64)  # 可选：时间戳
    #         ]
    #         schema = CollectionSchema(fields, "用户长期画像记忆")
    #         self.memory_collection = Collection(mem_name, schema, using="link")
    #
    #         # 创建索引
    #         index_params = {"metric_type": "IP", "index_type": "FLAT", "params": {"M": 8, "efConstruction": 64}}
    #         self.memory_collection.create_index("vector", index_params)
    #         self.memory_collection.load()
    #         print(f"🆕 [Memory] 新建长期记忆库: {mem_name}")
    # 修改 SenseVoice_Agent_Brain.py 中的 SmartAgentBrain 类

    async def update_memory_logic(self, new_fact):
        """
        核心记忆更新逻辑：检索 -> 比较 -> (删除旧的) -> 写入新的
        """
        if not self.memory_collection: return

        print(f"🤔 [Memory] 正在评估新记忆: {new_fact}")

        # 1. 先去搜一下有没有相关的旧记忆
        # 阈值设低一点(0.4)，确保能搜到相关的；太高可能漏掉矛盾点
        similar_memories = self.milvus.search_memory(new_fact, top_k=3)

        # 过滤掉相似度太低的，只保留真正相关的
        candidates = [m for m in similar_memories if m['score'] > 0.4]

        ids_to_delete = []

        # 2. 如果找到了相似记忆，需要 LLM 介入判断冲突
        if candidates:
            candidates_str = "\n".join([f"ID:{m['id']} 内容:{m['text']}" for m in candidates])

            check_prompt = f"""
            你是一个记忆管理员。请判断【新信息】与【已有记忆】的关系。

            【已有记忆】:
            {candidates_str}

            【新信息】:
            {new_fact}

            逻辑判断规则: 
            1. **冲突/修正**: 如果新信息与旧记忆矛盾（如“喜欢辣”变“不吃辣”），或者新信息是旧记忆的更新版本，输出 "DELETE: <旧记忆ID>"。
            2. **冗余**: 如果新信息在已有记忆中完全包含了，不需要重复记录，输出 "IGNORE"。
            3. **补充/无关**: 如果新信息是补充的新知识，与旧记忆不冲突，输出 "KEEP"。

            请只输出决策结果。如果有多个ID要删除，用逗号分隔。
            示例输出: "DELETE: 44213, 44215" 或 "IGNORE" 或 "KEEP"
            """

            try:
                check_res = await self.aclient.chat.completions.create(
                    model=self.LLM_MODEL,
                    messages=[{"role": "user", "content": check_prompt}],
                    temperature=0.0
                )
                decision = check_res.choices[0].message.content.strip()
                print(f"⚖️ [Memory] 记忆冲突裁决: {decision}")

                if "IGNORE" in decision:
                    print("🚫 [Memory] 信息冗余，跳过写入。")
                    return  # 直接结束，不写入

                if "DELETE:" in decision:
                    # 解析要删除的 ID
                    id_str = decision.split("DELETE:")[1].strip()
                    # 处理可能出现的非数字字符
                    import re
                    ids = re.findall(r'\d+', id_str)
                    ids_to_delete = [int(i) for i in ids]

            except Exception as e:
                print(f"❌ [Memory] 裁决过程出错: {e}")

        # 3. 执行操作
        # 如果有要删除的旧记忆，先删除
        if ids_to_delete:
            self.milvus.delete_memory_by_ids(ids_to_delete)

        # 写入新记忆 (使用原来的 insert 逻辑，但要确保调用的是 milvus 实例的方法)
        # 注意：这里调用的是 Milvus 类里的 insert 逻辑，或者你在这里手动 insert
        vec = self.milvus.embedding(new_fact)
        if vec:
            import time
            data = [[vec], [new_fact], [int(time.time())]]
            self.memory_collection.insert(data)
            # self.memory_collection.flush() # 频繁 flush 影响性能，可以累积或定时 flush
            print(f"💾 [Memory] 写入新记忆: {new_fact}")

    async def extract_and_save_memory(self, user_text):
        """[后台任务] 提取事实并触发更新流程"""
        prompt = f"""
        分析用户输入，提取用户的核心画像事实（喜好、习惯、身体状况、计划）。
        只提取**长期有效**的信息。
        如果无有效信息，输出 "NONE"。

        用户输入: "{user_text}"
        输出示例: "用户现在不吃辣了"
        """
        try:
            res = await self.aclient.chat.completions.create(
                model=self.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            fact = res.choices[0].message.content.strip()

            if fact and "NONE" not in fact and len(fact) > 2:
                # 改为调用新的更新逻辑
                await self.update_memory_logic(fact)

        except Exception as e:
            print(f"记忆提取错误: {e}")

    # def remember_fact(self, text):
    #     """写入记忆：将文本向量化并存入 Milvus"""
    #     if not self.memory_collection: return
    #
    #     vec = self.milvus.embedding(text)
    #     if vec:
    #         import time
    #         # 插入数据
    #         data = [
    #             [vec],  # vector
    #             [text],  # text
    #             [int(time.time())]  # timestamp
    #         ]
    #         self.memory_collection.insert(data)
    #         self.memory_collection.flush()  # 强制落盘
    #         print(f"💾 [Memory] 已记住: {text}")

    def recall_memories(self, query, top_k=2):
        """回忆：根据当前话题检索相关记忆"""
        if not self.memory_collection: return []

        vec = self.milvus.embedding(query)
        if not vec: return []

        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        try:
            res = self.memory_collection.search(
                data=[vec], anns_field="vector", param=search_params, limit=top_k,
                output_fields=["text"]
            )
            # 提取文本
            memories = [hit.entity.get("text") for hit in res[0] if hit.distance > 0.4]  # 阈值过滤，避免不相关的记忆
            if memories:
                print(f"💭 [Memory] 想起了: {memories}")
            return memories
        except Exception as e:
            print(f"❌ [Memory] 回忆失败: {e}")
            return []

    # async def extract_and_save_memory(self, user_text):
    #     """[后台任务] 使用 LLM 判断用户说的话是否包含值得记录的事实"""
    #     # 并不是每一句话都要记，只有包含“我...”的事实才值得记
    #     prompt = f"""
    #     分析用户的话，提取关于用户的核心事实（如喜好、关系、位置、计划等）。
    #     如果包含有价值的长期信息，请提取为简短的陈述句。
    #     如果没有（如仅仅是问候或闲聊），输出 "NONE"。
    #
    #     用户输入: "{user_text}"
    #
    #     输出示例:
    #     用户: "我不仅仅喜欢吃苹果，还对花生过敏" -> "用户喜欢吃苹果，且对花生过敏"
    #     用户: "今天天气不错" -> "NONE"
    #     """
    #     try:
    #         res = await self.aclient.chat.completions.create(
    #             model=self.LLM_MODEL,
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=0.1
    #         )
    #         fact = res.choices[0].message.content.strip()
    #         if fact and "NONE" not in fact and len(fact) > 2:
    #             self.remember_fact(fact)
    #     except Exception as e:
    #         print(f"记忆提取错误: {e}")



    # def connect_milvus(self):
    #     """连接 Milvus 数据库"""
    #     try:
    #         # 调试：打印连接参数
    #         print(f"🔧 [Debug] MILVUS_URI: {self.MILVUS_URI}")
    #         print(f"🔧 [Debug] MILVUS_TOKEN: {'已设置' if self.MILVUS_TOKEN else '未设置'}")
    #
    #         connections.connect(alias="link", uri=self.MILVUS_URI, token=self.MILVUS_TOKEN)
    #         print("✅ [Brain] Milvus 连接成功")
    #         # 假设集合已经存在 (由之前的脚本创建)
    #         self.collection = Collection(name="MilVus_test", using="link")
    #         self.collection.load()
    #     except Exception as e:
    #         print(f"⚠️ [Brain] Milvus 连接失败或集合不存在: {e}")
    #         self.collection = None

    # def get_embedding(self, text):
    #     """调用 Embedding API"""
    #     # api_key = "610764dc-5ee9-41b1-aac8-1c9728a1e5cf"
    #     # url = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    #     url = self.ARK_BASE_URL + '/embeddings'
    #     # url = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.ARK_API_KEY}"
    #     }
    #     data = {
    #         "input": [text],
    #         "model": self.EMBEDDING_MODEL,
    #         "embedding_dimension": 2560  # 确保跟你的模型一致
    #     }
    #     try:
    #         print("执行响应Embedding请求:")
    #         response = requests.post(url, headers=headers, json=data)
    #         result = response.json()
    #         return result['data'][0]["embedding"]
    #     except Exception as e:
    #         print(f"❌ [Brain] Embedding 失败: {e}")
    #         return None

    def search_food_db(self, query_text):
        """查询 Milvus 数据库"""
        if not self.collection:
            print("❌ [Brain] Milvus 集合不可用，无法查询数据库")
            return None

        print(f"🔍 [Brain] 正在查询数据库: {query_text}")
        vec = self.milvus.embedding(query_text)
        if not vec: return None

        search_params = {"metric_type": "IP", "params": {"ef": 128}}
        try:
            res = self.collection.search(
                data=[vec],
                anns_field="vector",
                param=search_params,
                limit=5,
                output_fields=["item_name"]
            )
            print("检索结果分析: ", res)
            if res and res[0]:
                return res[0][0].entity.get('item_name')
        except Exception as e:
            print(f"❌ [Brain] 检索出错: {e}")
        return None

    def search_web(self, query, max_results=2):
        """联网搜索（带重试）"""
        print(f"🌐 [Brain] 正在联网搜索: {query} ...")
        for attempt in range(3):
            try:
                results = list(self.ddgs.text(query, max_results=max_results))
                print("result:", results)
                if not results:
                    return ""
                return "\n".join([f"标题: {r['title']}\n摘要: {r['body']}" for r in results])
            except Exception as e:
                print(f"⚠️ [Brain] 搜索尝试 {attempt + 1} 失败: {e}")
                if attempt < 2:
                    import time
                    time.sleep(1)
        return ""

    # def search_web(self, query, max_results=2):
    #     """联网搜索"""
    #     print(f"🌐 [Brain] 正在联网搜索: {query} ...")
    #     try:
    #         results = list(self.ddgs.text(query, max_results=max_results))
    #         if not results: return ""
    #         return "\n".join([f"标题: {r['title']}\n摘要: {r['body']}" for r in results])
    #     except Exception as e:
    #         print(f"❌ [Brain] 搜索失败: {e}")
    #         return ""

    async def process_user_query(self, user_text):
        """
        核心处理流：意图识别 -> (查库 OR 联网 OR 闲聊) -> 生成回复
        """

        # --- 1. 回忆 (Long-Term Retrieval) ---
        related_memories = self.recall_memories(user_text)
        memory_str = ""
        if related_memories:
            memory_str = f"【已知用户信息】: {';'.join(related_memories)}"
            print(f"🧠 注入记忆上下文: {memory_str}")

        # 1. 意图路由 Prompt
        route_prompt = f"""
        请分析用户文本，返回 JSON 格式（不要Markdown）：
        1. Call_elm (bool): 是否想点外卖/询问菜品？
        2. Food_candidate (str): 具体菜名或口味需求，无则为空。
        3. Need_Search (str): 如需查询实时信息(新闻/天气/百科)请输出搜索关键词，否则为空。

        用户输入："{user_text}"

        示例：{{"Call_elm": true, "Food_candidate": "皮蛋粥", "Need_Search": ""}}
        示例：{{"Call_elm": false, "Food_candidate": "", "Need_Search": "北京天气"}}
        """

        try:
            # --- 第一步：路由决策 ---
            route_res = await self.aclient.chat.completions.create(
                model=self.LLM_MODEL,
                messages=[{"role": "user", "content": route_prompt}],
                temperature=0.1
            )
            raw_json = route_res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            intent = json.loads(raw_json)
            print(f"🧠 [Brain] 意图分析: {intent}")

            final_response = ""

            # --- 分支 A: 点餐业务 ---
            if intent.get("Call_elm"):
                food_name = intent.get("Food_candidate")
                matched = self.search_food_db(food_name)
                if matched:
                    final_response = f"找到啦！我们要不要来一份{matched}？"
                else:
                    final_response = f"抱歉，菜单里好像没有找到{food_name}，换个别的试试？"

            # --- 分支 B: 联网搜索 ---
            elif intent.get("Need_Search"):
                search_q = intent.get("Need_Search")
                search_ctx = self.search_web(search_q)
                print("搜索结果: ", search_ctx)

                # 联网回答也需要带上历史记忆（比如“北京天气”，记忆中有“用户怕冷”）
                gen_prompt = f"""
                                {memory_str}
                                基于搜索结果和用户记忆回答。
                                用户问题：{user_text}
                                搜索结果：{search_ctx}
                                """
                resp = await self.aclient.chat.completions.create(
                    model=self.LLM_MODEL,
                    messages=[{"role": "user", "content": gen_prompt}]
                )
                final_response = resp.choices[0].message.content

                # gen_prompt = f"基于搜索结果回答用户：{user_text}\n\n搜索结果：\n{search_ctx}"
                # resp = await self.aclient.chat.completions.create(
                #     model=self.LLM_MODEL,
                #     messages=[{"role": "user", "content": gen_prompt}]
                # )
                # final_response = resp.choices[0].message.content

            # --- 分支 C: 纯闲聊 ---
            else:
                system_msg = "你叫小千，是一个活泼可爱的语音助手，回答请简短。"
                if memory_str:
                    system_msg += f"\n{memory_str}\n请在聊天中自然运用这些信息，体现你记得用户。"

                messages = [{"role": "system", "content": system_msg}]
                messages.extend(self.history)  # 加入短期历史
                messages.append({"role": "user", "content": user_text})

                chat_res = await self.aclient.chat.completions.create(
                    model=self.LLM_MODEL,
                    messages=messages
                )
                final_response = chat_res.choices[0].message.content


                # chat_res = await self.aclient.chat.completions.create(
                #     model=self.LLM_MODEL,
                #     messages=[
                #         {"role": "system", "content": "你叫小千，是一个活泼可爱的语音助手，回答请简短(50字以内)。"},
                #         {"role": "user", "content": user_text}
                #     ]
                # )
                # final_response = chat_res.choices[0].message.content
            # --- 收尾工作 ---
            # 1. 更新短期记忆
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": final_response})
            if len(self.history) > self.max_history_len:
                self.history = self.history[-self.max_history_len:]

            # 2. [异步] 提取并保存新记忆
            # 使用 asyncio.create_task 让它在后台运行，不阻塞当前的语音播报
            asyncio.create_task(self.extract_and_save_memory(user_text))

            return final_response

        except Exception as e:
            print(f"❌ [Brain] 处理异常: {e}")
            return "不好意思，我的大脑刚刚短路了一下，能再说一遍吗？"


# 测试代码
if __name__ == "__main__":
    brain = SmartAgentBrain()
    print(asyncio.run(brain.process_user_query("我想吃三鲜乌冬面")))
    print(asyncio.run(brain.process_user_query("今天北京天气怎么样")))