# 🎙️ Identity-Aware Multimodal Voice Agent (M-RAG-Voice)

> 一个具备**声纹身份感知**、**动态长期记忆**和**端云混合推理**能力的智能语音助手框架。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Milvus](https://img.shields.io/badge/VectorDB-Milvus-orange) ![ASR](https://img.shields.io/badge/ASR-SenseVoice-green) ![LLM](https://img.shields.io/badge/LLM-DeepSeek%2FQwen-blueviolet)

## 📖 项目简介

这是一个探索性的多模态语音交互系统。不同于传统的语音助手，该项目集成了**声纹识别 (Speaker Verification)** 和 **RAG (检索增强生成)** 技术。它不仅能听懂“你在说什么”，还能识别“你是谁”，并根据不同用户的身份调用专属的长期记忆库（如饮食习惯、历史偏好），提供高度个性化的回答。

### ✨ 核心特性

* **👥 多用户声纹识别**: 集成 CAM++ 模型，支持 1:N 声纹匹配。自动区分“主人”与“访客”，支持语音指令注册新用户。
* **🧠 动态进化记忆**: 基于 Milvus 向量数据库构建用户画像。具备“冲突裁决”机制，自动利用 LLM 分析新旧记忆冲突，实现记忆的自我更新与修正。
* **⚡ 端云混合架构**:
    * **端侧 (Local)**: 运行高频、低延迟任务（VAD, ASR-SenseVoice, SV-CAM++, TTS）。
    * **云侧/端侧灵活性**: LLM (DeepSeek/Qwen) 支持本地部署或 API 调用，平衡隐私与性能。
* **🛠️ 智能意图路由**: 能够区分闲聊、点餐（查询本地知识库）、联网搜索（Tavily）和系统指令。

## 🏗️ 系统架构

系统采用 **Audio-Text-Audio** 闭环架构，并嵌入了身份（Identity）层：
<img width="1408" height="768" alt="Gemini_Generated_Image_vvj5vovvj5vovvj5(1)" src="https://github.com/user-attachments/assets/3611130c-f044-4d2b-95f2-a3d07da6e85c" />




## 📂 项目结构
```text
.
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
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# 安装依赖
pip install -r requirements.txt
```

推荐依赖 (参考): funasr, modelscope, pymilvus, openai, webrtcvad, pyaudio, pygame, edge-tts, tavily-python, transformers, torch


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

# Milvus 向量数据库
URL=your_milvus_uri
Token=your_milvus_token

# 联网搜索工具
trivily_key=your_tavily_key
```


4. 运行
启动主程序：
```python
python SenseVoice_Agent_Main.py
```


## 💡 使用指南



```text
1.初次运行: 如果声纹库为空，系统会提示你进行注册。请根据语音提示录入“主人”的声音。

2.唤醒: 对着麦克风说 “小明同学” (Xiao Ming Tong Xue) 即可唤醒助手。

3.功能示例:
  * 记忆存储: "我以后不吃辣了，记住哦。" -> (系统更新数据库，删除旧的喜辣记忆)
  * 个性化问答: "我今天中午吃什么好？" -> (系统检索你的历史口味推荐)
  * 声纹注册: "我是张三，把我的声音录进去。" -> (触发注册流程)
```

## ⚠️ 注意事项


硬件要求: 推荐使用 `NVIDIA GPU` 运行 `ASR` 和`声纹模型`以获得最佳延迟体验。`Local_Model.py` 会自动检测 CUDA。

路径配置: 请务必检查 `Local_Model.py` 和 `SpeakerManager.py` 中的绝对路径，确保其在你的机器上有效。

## 🤝 贡献

欢迎提交 Issue 和 PR！如果你有更好的记忆管理策略或更轻量的模型实现，请随时分享。

## 🙏 Acknowledgments
ASR-LLM-TTS (https://github.com/ABexit/ASR-LLM-TTS)
