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
from tavily import TavilyClient
from Local_Model import Load_Model
import threading
# 加载环境变量(.env文件)
load_dotenv()
# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'


class SmartAgentBrain:
    def __init__(self, LOCAL_LLM=False):
        # ================= 配置区域 (请修改这里) =================
        self.ARK_API_KEY = os.environ.get("ARK_API_KEY")  # 填入你的 API Key
        self.ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.LLM_MODEL = "deepseek-ai/DeepSeek-V3"  # 你的推理模型 ID
        self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"  # 你的 Embedding 模型 ID

        self.MILVUS_URI = os.environ.get("URL")  # Milvus 地址
        self.MILVUS_TOKEN = os.environ.get("Token")  # Milvus Token

        self.search_web_key = os.environ.get("trivily_key")
        # =======================================================

        # 初始化 LLM 客户端
        self.aclient = AsyncOpenAI(
            api_key=self.ARK_API_KEY,
            base_url=self.ARK_BASE_URL,
        )

        # 检索工具
        self.client = TavilyClient(self.search_web_key)
        # self.connect_milvus()

        self.history = []  # 短期历史记录窗口
        self.max_history_len = 6  # 只保留最近6轮对话

        # 初始化类
        self.milvus = MilvusClass()
        self.milvus.connect_milvus()
        self.memory_collection = self.milvus.memory_collection
        self.collection = self.milvus.food_collection

        # 初始化本地类
        self.local_model  = Load_Model()
        self.LOCAL_LLM = LOCAL_LLM

    async def update_memory_logic(self, new_fact, user_id):
        """
        核心记忆更新逻辑：检索 -> 比较 -> (删除旧的) -> 写入新的
        """
        if not self.memory_collection: return

        print(f"[Memory] 正在评估新记忆: {new_fact}")

        # 1. 先去搜一下有没有相关个人的旧记忆
        similar_memories = self.milvus.search_memory(new_fact, user_id=user_id, top_k=3)
        # similar_memories = self.milvus.search_memory(new_fact, top_k=3)

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
                messages = [{"role": "user", "content": check_prompt}]
                if self.LOCAL_LLM:
                    # 基于本地部署的LLM
                    check_res = self.local_model.llm_chat(messages)
                    decision = check_res.strip()
                else:
                    # 基于API实现
                    check_res = await self.aclient.chat.completions.create(
                        model=self.LLM_MODEL,
                        messages=messages,
                        temperature=0.0
                    )
                    decision = check_res.choices[0].message.content.strip()
                print(f"[Memory] 记忆冲突裁决: {decision}")

                if "IGNORE" in decision:
                    print("[Memory] 信息冗余，跳过写入。")
                    return  # 直接结束，不写入

                if "DELETE:" in decision:
                    # 解析要删除的 ID
                    id_str = decision.split("DELETE:")[1].strip()
                    # 处理可能出现的非数字字符
                    import re
                    ids = re.findall(r'\d+', id_str)
                    ids_to_delete = [int(i) for i in ids]

            except Exception as e:
                print(f"[Memory] 裁决过程出错: {e}")

        # 3. 执行操作
        # 如果有要删除的旧记忆，先删除
        if ids_to_delete:
            print(f"[Memory] 准备删除冲突记忆: {ids_to_delete} (所属用户: {user_id})")
            self.milvus.delete_memory_by_ids(ids_to_delete, user_id)

        # 写入新记忆 (使用原来的 insert 逻辑，但要确保调用的是 milvus 实例的方法)
        # 注意：这里调用的是 Milvus 类里的 insert 逻辑，或者你在这里手动 insert
        vec = self.milvus.embedding(new_fact)
        if vec:
            import time

            self.milvus.insert_memory(new_fact, user_id)

            print(f"[Memory] 写入新记忆: {new_fact}, 关于用户 {user_id}")

    async def extract_and_save_memory(self, user_text, user_id):
        """[后台任务] 提取事实并触发更新流程"""
        print(f"开始处理用户输入: {user_text}")
        prompt = f"""
        ### 任务：提取用户长期有效核心画像信息
        1.  **提取范围（仅以下4类长期信息）**
            - 喜好：长期稳定的兴趣、饮食偏好、爱好（例：喜欢吃甜食、长期喜欢打篮球）
            - 习惯：长期保持的行为模式（例：每天早上跑步、习惯早睡早起）
            - 身体状况：长期稳定的健康状态（例：对牛奶过敏、有慢性咽炎）
            - 计划：长期规划或固定安排（例：打算今年考驾照、计划每月读两本书）
        2.  **排除条件（这些信息不提取）**
            - 短期临时需求：比如"今天想吃火锅"、"明天要去逛街"
            - 一次性行为：比如"昨天看了电影"、"上周买了衣服"
            - 无明确长期属性的内容：比如"这个电影很好看"、"今天天气不错"
        3.  **输出规则**
            - 仅提取用户明确表述的**长期有效**信息，不做任何推测、扩展、总结
            - 信息必须直接来自用户输入，不能添加额外词汇
            - 格式：用陈述句直接描述，语言简洁
            - 如果没有符合要求的长期信息，**严格输出大写的 NONE**，不要输出其他任何内容
        4.  **示例参考**
            示例1：
            用户输入: "我对海鲜过敏，一直都不能吃"
            输出: 用户对海鲜过敏
            示例2：
            用户输入: "我每天都要喝一杯咖啡，这是多年的习惯"
            输出: 用户每天喝一杯咖啡
            示例3：
            用户输入: "我今天想吃麻辣烫"
            输出: NONE
            示例4：
            用户输入: "我打算明年去国外留学，已经规划很久了"
            输出: 用户打算明年去国外留学
            示例5：
            用户输入: "昨天我去跑步了"
            输出: NONE

        ### 用户输入
        "{user_text}"

        ### 最终输出
        """
        try:

            messages = [{"role": "user", "content": prompt}]
            if self.LOCAL_LLM:
                # 基于本地LLM
                print("本地LLM")
                res = self.local_model.llm_chat(messages)
                print("提取记忆结果: ", res)
                fact = res.strip()
            else:
                # 基于API
                print("基于API")
                res = await asyncio.wait_for(
                    self.aclient.chat.completions.create(
                        model=self.LLM_MODEL,
                        messages=messages,
                        temperature=0.1
                    ),
                    timeout=30.0  # 30秒超时
                )
                # res = await self.aclient.chat.completions.create(
                #     model=self.LLM_MODEL,
                #     messages=messages,
                #     temperature=0.1
                # )
                fact = res.choices[0].message.content.strip()
            print("fact: ", fact)
            if fact and "NONE" not in fact:
                await self.update_memory_logic(fact, user_id)
        except Exception as e:
            print(f"记忆提取错误: {e}")

    def recall_memories(self, query, user_id, top_k=2):
        # 传递 user_id 给 milvus
        results = self.milvus.search_memory(query, user_id=user_id, top_k=top_k)
        memories = [item["text"] for item in results]
        return memories


    def search_food_db(self, query_text):
        """查询 Milvus 数据库"""
        if not self.collection:
            print("[Brain] Milvus 集合不可用，无法查询数据库")
            return None

        print(f"[Brain] 正在查询数据库: {query_text}")
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
            print(f"[Brain] 检索出错: {e}")
        return None

    def search_web(self, query, max_results=2):
        """联网搜索（带重试）"""
        print(f"[Brain] 正在联网搜索: {query} ...")
        for attempt in range(3):
            try:
                response = self.client.search(
                    query=query,
                    search_depth="advanced"
                )
                results = response["results"][:max_results]
                print("result:", results)
                if not results:
                    return ""
                return "\n".join([f"标题: {r['title']}\n摘要: {r['content']}" for r in results])
            except Exception as e:
                print(f"[Brain] 搜索尝试 {attempt + 1} 失败: {e}")
                if attempt < 2:
                    import time
                    time.sleep(1)
        return ""


    async def process_user_query(self, user_text, user_id):
        """
            意图识别 -> (查库 OR 联网 OR 闲聊) -> 生成回复
        """
        print(f"[Brain] 处理请求，当前用户: {user_id}")

        # --- 回忆特定用户记忆成分 ---
        related_memories = self.recall_memories(user_text, user_id)
        memory_str = ""
        if related_memories:
            memory_str = f"【关于 {user_id} 的记忆】: {';'.join(related_memories)}"

        # 意图路由 Prompt, 针对小模型的优化 Prompt, 增强其指令遵循的能力
        route_prompt = f"""
        ### 角色设定
        你是一个意图识别API。仅输出JSON格式结果，不要包含Markdown标记或其他文字。

        ### 字段定义
        - Call_elm (bool): 是否包含点餐/外卖意图(想吃、点菜等)。
        - Food_candidate (str): 提取具体的菜名/食物。
        - Need_Search (str): 提取需要联网搜索的关键词(天气/百科/价格等)。
        - Register_Action (str): 提取注册意图。若包含名字提取名字(如"我是张三"); 若想注册但未提供名字输出"Unknown_User"; 否则为空。

        ### 参考示例 (请严格模仿以下格式)
        输入: "我想吃皮蛋瘦肉粥"
        JSON: {{"Call_elm": true, "Food_candidate": "皮蛋瘦肉粥", "Need_Search": "", "Register_Action": ""}}

        输入: "查询明天北京的天气"
        JSON: {{"Call_elm": false, "Food_candidate": "", "Need_Search": "明天北京天气", "Register_Action": ""}}

        输入: "我要注册新用户"
        JSON: {{"Call_elm": false, "Food_candidate": "", "Need_Search": "", "Register_Action": "Unknown_User"}}

        输入: "我是张三，把我的声音录进去"
        JSON: {{"Call_elm": false, "Food_candidate": "", "Need_Search": "", "Register_Action": "张三"}}

        输入: "随便聊聊"
        JSON: {{"Call_elm": false, "Food_candidate": "", "Need_Search": "", "Register_Action": ""}}

        ### 当前任务
        输入: "{user_text}"
        JSON: """

        try:
            # --- 第一步：路由决策 ---
            messages = [{"role": "user", "content": route_prompt}]
            if self.LOCAL_LLM:
                # 本地部署llm
                print("本地部署调用")
                route_res = self.local_model.llm_chat(messages)
                print("route_res:", route_res)
                raw_json = route_res.replace("```json", "").replace("```", "").strip()
            else:
                # API接口
                print("执行api调用")
                route_res = await self.aclient.chat.completions.create(
                    model=self.LLM_MODEL,
                    messages=messages,
                    temperature=0.1
                )
                raw_json = route_res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            print("raw_json:", raw_json)
            intent = json.loads(raw_json)
            print(f"[Brain] 意图分析: {intent}")

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
                ### 任务要求
                你需要结合【用户记忆】和【搜索结果】，回答用户问题，满足以下硬性规则：
                1.  **极致简短**：回答控制在2句话以内，总字数不超过30字，只保留核心信息，去掉所有修饰词
                2.  **口语化**：用日常说话的语气，适合语音播报，避免书面语、专业术语
                3.  **优先级**：优先使用【搜索结果】的信息；搜索结果无相关内容时，再用【用户记忆】；两者都无则回复"暂无相关信息"
                4.  **禁止内容**：不解释、不补充、不扩展，不出现括号、引号等特殊符号

                ### 【用户记忆】
                你正在和 {user_id} 对话。
                {memory_str}

                ### 【搜索结果】
                {search_ctx}

                ### 用户问题
                {user_text}

                ### 示例参考
                示例1：
                用户记忆：用户喜欢喝拿铁
                搜索结果：今天咖啡店拿铁买一送一
                用户问题：今天喝咖啡有优惠吗
                输出：今天咖啡店拿铁买一送一

                示例2：
                用户记忆：用户对芒果过敏
                搜索结果：（空）
                用户问题：我能吃芒果吗
                输出：你对芒果过敏，不能吃

                示例3：
                用户记忆：（空）
                搜索结果：（空）
                用户问题：明天会下雨吗
                输出：暂无相关信息

                ### 最终输出
                （仅输出回答内容，无其他文字）
                """
                messages = [{"role": "user", "content": gen_prompt}]
                if self.LOCAL_LLM:
                    # 本地部署llm方式
                    resp = self.local_model.llm_chat(messages)
                    print("resp:", resp)
                    final_response = resp
                else:
                    # 基于API的调用方式
                    resp = await self.aclient.chat.completions.create(
                        model=self.LLM_MODEL,
                        messages=messages
                    )
                    final_response = resp.choices[0].message.content

            # --- 分支 C: 注册声纹 ---
            elif intent.get("Register_Action"):
                target_name = intent.get("Register_Action")
                # 返回特殊标记给 Main 函数处理
                return f"ACTION_REGISTER:{target_name}"
            else:
                system_msg = f"你叫小千，是一个活泼可爱的语音助手，现在的对话者是 {user_id},回答请简短。"
                if memory_str:
                    system_msg += f"\n{memory_str}\n请在聊天中自然运用这些信息，体现你记得用户。"

                messages = [{"role": "system", "content": system_msg}]
                messages.extend(self.history)  # 加入短期历史
                messages.append({"role": "user", "content": user_text})

                if self.LOCAL_LLM:
                    # 基于本地LLM实现
                    chat_res = self.local_model.llm_chat(messages)
                    final_response = chat_res
                else:
                    # 基于API接口
                    chat_res = await self.aclient.chat.completions.create(
                        model=self.LLM_MODEL,
                        messages=messages
                    )
                    final_response = chat_res.choices[0].message.content

            # 更新短期记忆
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": final_response})
            if len(self.history) > self.max_history_len:
                self.history = self.history[-self.max_history_len:]
            print("执行异步操作...")
            # 提取并保存新记忆, 使用asyncio.create_task会造成冲突, 最后的任务执行被跳过
            # 是因为asyncio.run(process_user_query()) 创建了一个新的、临时的事件循环, 当return之后, 主函数执行完毕, 然后就会强制终止时间循环


            # 定义一个同步的包装函数，因为 threading target 需要同步函数, 实现并行操作
            def run_memory_task_sync(text, uid):
                # 在新线程中创建一个新的事件循环来运行该异步任务
                try:
                    asyncio.run(self.extract_and_save_memory(text, uid))
                except Exception as e:
                    print(f"后台记忆任务出错: {e}")

            # 启动守护线程 (Daemon Thread)
            # daemon=True 意味着如果主程序退出了，这个线程也会随之退出，不会卡死进程
            t = threading.Thread(target=run_memory_task_sync, args=(user_text, user_id), daemon=True)
            t.start()

            return final_response

        except Exception as e:
            print(f"处理异常: {e}")
            return "不好意思，我的大脑刚刚短路了一下，能再说一遍吗？"


# 测试代码
if __name__ == "__main__":
    brain = SmartAgentBrain()
    # brain.local_model.llm_chat(messages = [{"role": "user", "content": "明天北京天气怎么样？"}])
    asyncio.run(brain.extract_and_save_memory("明天北京天气怎么样？", "主人"))
    # brain.extract_and_save_memory("明天北京天气怎么样？")
    # print("result: ", res)
    # print(asyncio.run(brain.process_user_query("我想吃三鲜乌冬面")))
    # print(asyncio.run(brain.process_user_query("明天北京天气怎么样")))