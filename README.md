# 🎙️ Identity-Aware Multimodal Voice Agent (M-RAG-Voice)

> 一个具备**声纹身份感知**、**动态长期记忆**和**端云混合推理**能力的智能语音助手框架。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Milvus](https://img.shields.io/badge/VectorDB-Milvus-orange) ![ASR](https://img.shields.io/badge/ASR-SenseVoice-green) ![LLM](https://img.shields.io/badge/LLM-DeepSeek%2FQwen-blueviolet)

## 📖 项目简介

这是一个探索性的多模态语音交互系统。不同于传统的语音助手，该项目集成了**声纹识别 (Speaker Verification)** 和 **RAG (检索增强生成)** 技术。它不仅能听懂“你在说什么”，还能识别“你是谁”，并根据不同用户的身份调用专属的长期记忆库（如饮食习惯、历史偏好），提供高度个性化的回答。

### ✨ 核心特性

* **👥 多用户声纹识别**: 集成 CAM++ 模型，支持 1:N 声纹匹配。自动区分“主人”与“访客”，支持语音指令注册新用户。
* **🧠 动态进化记忆**: 基于 Milvus 向量数据库构建用户画像。具备“冲突裁决”机制，自动利用 LLM 分析新旧记忆冲突，实现记忆的自我更新与修正。
    * **本地部署Neo4j数据库**: 用于存储结构化的长期记忆和知识图谱。
    * **本地部署MilVus数据库**: 用于存储向量化的记忆片段，支持高效检索。
    * **与Mem0的区别:**: Mem0实现知识图谱时侧重插入，不强调数据一致性（向量库和知识图谱库），本项目强调数据一致性，通过冲突裁决机制确保长期记忆的准确性和可靠性。
* **⚡ 端云混合架构**:
    * **端侧 (Local)**: 运行高频、低延迟任务（VAD, ASR-SenseVoice, SV-CAM++, TTS）。
    * **云侧/端侧灵活性**: LLM (DeepSeek/Qwen) 支持本地部署或 API 调用，平衡隐私与性能。
* **🛠️ 智能意图路由**: 能够区分闲聊、点餐（查询本地知识库）、联网搜索（Tavily）和系统指令。
* **🔧 大模型微调**: 使用CARMEM车载对话数据集进行多轮对话微调Qwen3-4B-Instruct-2507。

## 🏗️ 系统架构

系统采用 **Audio-Text-Audio** 闭环架构，并嵌入了身份（Identity）层：
<img width="1408" height="768" alt="Gemini_Generated_Image_vvj5vovvj5vovvj5(1)" src="https://github.com/user-attachments/assets/3611130c-f044-4d2b-95f2-a3d07da6e85c" />


### Neo4j数据库
<img width="500" height="500" alt="Gemini_Generated_Image_vvj5vovvj5vovvj5(1)" src="fig/neo4j.png" />


## 📋 项目任务清单
### ✅ 已完成
- [x] 多用户自动声纹识别
- [x] 长短期记忆库，短期记忆（依赖Memory）, 长期记忆（依赖Milvus）
- [x] 人声检测和播报互斥机制
- [x] 用户偏好的动态更新
- [x] 本地端侧部署与推理
- [x] 编写 README.md 基础文档与环境配置说明
- [x] 基于知识图谱的长期记忆（本地部署的Neo4j图数据库）
- [x] 微调本地模型构建个性化模型-用于智能座舱交互
- [x] 实时语音播报（边生成边播报）
### 📌 待完成




## 📂 项目结构
```text
.
├── Agent_ASR_V1(history)      # 历史版本(未实现知识图谱抽取与存储)
├── Agent_ASR_V2(history)      # 历史版本(实现了知识图谱抽取与存储，未引入实时播报)
├── Fine-Tuning                # 大模型微调相关代码（Qwen3-4B-Instruct-2507）
├── fig                        # 项目相关图片资源
├── LLM deployment             # 大模型本地部署: Xinference/VLLM/Transformers/Ollama
├── Knowledge_Graph.py         # 知识图谱处理：负责从文本中抽取实体与关系，构建 Neo4j 图数据库
├── Local_Model.py             # 模型加载器：单例模式管理 LLM, ASR, CAM++ 模型的加载
├── SpeakerManager.py          # 身份管理：处理声纹注册、加载与 1:N 匹配逻辑
├── SenseVoice_Agent_Brain.py  # 核心大脑：负责意图路由、RAG 检索、记忆提取与冲突更新
├── SenseVoice_Agent_Main.py   # 主程序：处理音频 I/O, VAD, 多线程调度与全流程控制
├── Milvus.py                  # 数据层：封装 Milvus 向量库的增删改查与 Embedding 操作
└── requirements.txt           # 项目依赖
```

## 🚀 快速开始
1. 环境准备
确保你已安装 Python 3.10+，并拥有一个运行中的 Milvus 实例（Docker 或 Cloud）。
```bash
# 克隆仓库
git clone https://github.com/CHM00/Agent_ASR.git
cd Agent_ASR

# 安装依赖
pip install -r requirements.txt
```

推荐依赖: `funasr, modelscope, pymilvus, openai, webrtcvad, pyaudio, pygame, edge-tts, tavily-python, transformers, torch`


2. 模型下载与配置
请修改 Local_Model.py 中的模型路径，指向你本地下载的模型权重：
```python
# Local_Model.py
self.llm_model_path = r"path/to/your/Qwen"  # LLM 模型路径
self.funasr_model_path = r"path/to/SenseVoice" # ASR 模型路径
self.CAM_model_path = r"path/to/CAM++"      # 声纹模型路径
```


3. 环境变量设置
在项目根目录创建 .env 文件，填入必要的 API Key 和数据库配置：
```text
# LLM API (如使用 DeepSeek/Volcengine)
ARK_API_KEY=your_api_key
ARK_BASE_URL=[https://ark.cn-beijing.volces.com/api/v3](https://ark.cn-beijing.volces.com/api/v3)

# 本地部署的 Milvus 向量数据库
URL=your_milvus_uri

# 联网搜索工具
trivily_key=your_tavily_key

# 本地部署 Neo4j 数据库
NEO4J_URI=bolt://47.113.202.238:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=Zchm0903
```


4. 运行
启动主程序：
```python
python SenseVoice_Agent_Main.py
```


## 语音合成方案性能对比分析 (Benchmark)

本项目对三种主流的语音合成（TTS）方案进行了实机测试，分别代表了 **在线 API**、**本地大模型** 和 **本地轻量级引擎** 三种技术路线。

> **测试环境架构**：
> * **LLM**: DeepSeek-V3 (API) + 联网搜索
> * **ASR**: SenseVoice (本地)
> * **硬件**: 单显卡工作站
> 
> 

### 📊 1. 核心指标对比总结

| 方案名称 | **pyttsx3** | **Edge-TTS** | **CosyVoice** |
| --- | --- | --- | --- |
| **类型** | 本地系统引擎 (SAPI5/nsss) | 在线 API (微软 Azure) | 本地生成式大模型 (AI) |
| **首字延迟 (TTFA)** | **⚡ 极速 (< 0.5s)** | 🚀 较快 (1.5s - 3s) | 🐢 较慢 (5s - 11s)* |
| **端到端总延迟** | **7s - 9s** | 9s - 12s | 14s - 25s |
| **系统资源占用** | **极低 (CPU)** | 低 (仅需联网) | **极高 (强依赖 GPU)** |
| **对流程影响** | **无阻塞 (极致流畅)** | 轻微网络依赖 | **严重阻塞 (GPU 计算排队)** |
| **语音自然度** | ⭐⭐ (机械感) | ⭐⭐⭐⭐ (自然流畅) | ⭐⭐⭐⭐⭐ (真人情感级) |
| **适用场景** | 调试、极速交互 | 兼顾速度与音质 | 离线数字人、高拟真 |

> **注***：在使用 DeepSeek API 的架构下，CosyVoice 的高延迟主要源于其**自身的推理计算耗时**以及**GPU 资源的调度开销**。

---

### ⏱️ 2. 阶段耗时分析

我们对一次完整的交互流程（语音输入 -> ASR -> LLM 思考 -> TTS 播报）进行了毫秒级埋点分析。

#### 🟢 方案 A: pyttsx3

* **ASR (听)**: `0.2s - 0.7s` (SenseVoice)
* **LLM (想)**: `6.0s - 8.0s` (DeepSeek-V3 API + 联网搜索)
* *分析*: 此时的延迟完全来自于**网络 IO**（搜索耗时 + API请求耗时），本地计算没有任何拖累。


* **TTS (说)**: `0.1s - 0.5s`
* *分析*: 几乎瞬间完成，完全消除了合成等待时间。


* **总延迟**: **~7.5 秒**

#### 🔵 方案 B: Edge-TTS

* **ASR (听)**: `0.5s - 1.0s`
* **LLM (想)**: `6.0s - 9.0s`
* **TTS (说)**: `1.5s - 3.0s`
* *分析*: 依赖微软服务器的响应速度，受网络波动影响，且需要等待音频下载。


* **总延迟**: **~10.5 秒**

#### 🔴 方案 C: CosyVoice

* **ASR (听)**: `0.2s - 0.8s`
* **LLM (想)**: `6.0s - 9.0s`
* *注意*: 虽然 LLM 在云端，但复杂的本地 GPU 调度可能会间接影响 Python 主线程的效率。


* **TTS (说)**: `5.0s - 11.0s` 🔺
* *分析*: 这是最大的瓶颈。CosyVoice 需要进行繁重的 GPU 推理生成音频，且目前采用非流式合成（整句生成），导致用户必须等待整句合成完毕才能听到声音。


* **总延迟**: **15s - 25s**

---

### 3. 性能瓶颈深度分析

#### 现象：为什么总延迟依然有 7 秒以上？

即使使用了秒回的 `pyttsx3`，系统依然有 7-8 秒的思考时间。

* **原因**: **串行网络链路 (Serial Network Chain)**。
目前的处理流程是串行的：`意图识别(API)` -> `联网搜索(Web IO)` -> `最终生成(API)`。
* **搜索耗时**: Tavily/DuckDuckGo 的爬取通常需要 3-4 秒。
* **API 耗时**: DeepSeek 处理长文本（搜索结果）的 Prefill 和传输需要 2-3 秒。


* **优化方向**: 必须实施 **搜索结果截断**（减少 API 处理量）和 **意图识别本地化**（减少一次 API 调用），目标可将延迟降至 3-4 秒。

---

## 💡 使用指南

* **初次运行**: 如果声纹库为空，系统会提示你进行注册。请根据语音提示录入“主人”的声音。
* **唤醒**: 对着麦克风说 “小明同学” (Xiao Ming Tong Xue) 即可唤醒助手。
* **功能示例**:
  * **记忆存储**: "我以后不吃辣了，记住哦。" -> (系统更新数据库，删除旧的喜辣记忆)
  * **个性化问答**: "我今天中午吃什么好？" -> (系统检索你的历史口味推荐)
  * **声纹注册**: "我是张三，把我的声音录进去。" -> (触发注册流程)


## 测试记忆提取能力
* **用户喜好反转识别**: 旧记忆:我超级喜欢吃香菜，每顿饭都要加。新信息: 我现在一点都不吃香菜了，那个味道太恶心了。
* **状态更新识别**: 旧记忆: 我还在上大学，是个学生。新内容: 我终于工作了，现在是一名程序员。
* **逻辑冲突**: 旧记忆: 我对海鲜严重过敏，吃一点就会起疹子。新内容: 今晚的麻辣小龙虾真好吃，我还想再吃一斤。
* **单一属性覆盖（凉/热）**: 旧记忆: 我只喝热水，从来不喝冰的。新内容: 给我来一杯冰美式，越冰越好，我现在喜欢喝凉的。

### 发现问题：
* （1）**对于地点状态不敏感**: 地点冲突识别不理想（先存了“我在上海定居”，输入“我在杭州定居”未识别出冲突）
* （2）**逻辑冲突识别不够完善**:（先存了“海鲜过敏”，输入“喜欢小龙虾”，语义检索未召回）
* （3）**身份变更不敏感**：程序员和学生身份转换识别不够理想程序员和学生身份转换识别不够理想
* （4）**冗余处理**: 对于“我不吃香菜了”和“我讨厌香菜了”两条信息，均会存为新记忆，未能识别冗余。

### 解决方案：
（1）（3）prompt修改: 明确规则定义
（2）检索优化: 提高召回数量，扩大检索范围，并增加基于关键词的检索
（4）新增冗余操作: 在冲突裁决前，先进行冗余检测，若新旧记忆相似度过高，则判定为冗余，拒绝存储。

## ⚠️ 注意事项

* **硬件要求**: 推荐使用 `NVIDIA GPU` 运行 `ASR` 和`声纹模型`以获得最佳延迟体验。`Local_Model.py` 会自动检测 CUDA。
* **路径配置**: 请务必检查 `Local_Model.py` 和 `SpeakerManager.py` 中的绝对路径，确保其在你的机器上有效。

## 🤝 贡献

欢迎提交 Issue 和 PR！如果你有更好的记忆管理策略或更轻量的模型实现，请随时分享。

## 🙏 Acknowledgments
ASR-LLM-TTS (https://github.com/ABexit/ASR-LLM-TTS)
