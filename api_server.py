"""
FastAPI Server for AgentASR_HTTP
提供语音识别、文本对话、语音合成、声纹管理等 Web 服务
"""
import os
import io
import json
import tempfile
import asyncio
import re
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles

# 导入本地模块
from SenseVoice_Agent_Brain import SmartAgentBrain
from SpeakerManager import SpeakerManager

# 全局配置
UPLOAD_DIR = "./uploads"
SPEAKER_DIR = "./SpeakerVerification_DIR/users"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

# 全局单例
agent_brain: Optional[SmartAgentBrain] = None
speaker_manager: Optional[SpeakerManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型"""
    global agent_brain, speaker_manager

    print("========================================")
    print("🚀 AgentASR FastAPI Server 启动中...")
    print("========================================")

    # 启动时初始化 SmartAgentBrain
    print("⏳ 正在加载模型...")
    agent_brain = SmartAgentBrain(LOCAL_LLM=False)
    print("✅ SmartAgentBrain 初始化完成")

    # 初始化声纹管理器
    speaker_manager = SpeakerManager(
        SPEAKER_DIR,
        agent_brain.local_model.CAM_model,
        threshold=0.35
    )
    print("✅ SpeakerManager 初始化完成")

    print("========================================")
    print("🎉 服务启动完成！")
    print(f"📡 API 文档: http://localhost:8000/docs")
    print(f"🌐 前端界面: http://localhost:8000/static/")
    print("========================================")

    yield

    # 关闭时清理
    print("🛑 服务关闭中...")
    if agent_brain and agent_brain.aclient:
        await agent_brain.aclient.close()
    print("✅ 清理完成")


# 创建 FastAPI 应用
app = FastAPI(
    title="AgentASR HTTP API",
    description="智能语音助手 Web 服务 - 支持 ASR、LLM 对话、TTS、声纹识别",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")


# ==================== 工具函数 ====================

async def get_brain() -> SmartAgentBrain:
    """依赖注入：获取 agent_brain 实例"""
    if agent_brain is None:
        raise HTTPException(status_code=503, detail="模型尚未初始化")
    return agent_brain


async def get_speaker_manager() -> SpeakerManager:
    """依赖注入：获取 speaker_manager 实例"""
    if speaker_manager is None:
        raise HTTPException(status_code=503, detail="声纹管理器尚未初始化")
    return speaker_manager


async def run_asr(audio_path: str) -> str:
    """异步执行 ASR 识别"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: agent_brain.local_model.funasr_model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=False
        )
    )


async def run_tts(text: str) -> bytes:
    """异步执行 TTS 合成 (使用 pyttsx3)"""
    def synthesize_sync(txt: str) -> bytes:
        import pyttsx3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            engine = pyttsx3.init()
            engine.save_to_file(txt, temp_path)
            engine.runAndWait()
            del engine

            # 读取生成的音频文件
            with open(temp_path, "rb") as af:
                return af.read()
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, synthesize_sync, text)


async def stream_chat_response(user_id: str, user_text: str):
    """生成流式聊天响应"""
    try:
        full_text = ""
        async for chunk in agent_brain.process_user_query(user_text, user_id):
            if chunk.startswith("ACTION_REGISTER:"):
                # 特殊指令：注册声纹
                yield f"data: {json.dumps({'type': 'action', 'action': 'register', 'target': chunk.split(':')[1]})}\n\n"
                return

            print("生成文本片段:", chunk)  # 调试输出
            # 普通文本片段
            yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
            full_text += chunk

        # 流式结束标记
        yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ==================== API 端点 ====================

@app.get("/")
async def root():
    """根路径：重定向到前端"""
    return {"message": "AgentASR HTTP API", "docs": "/docs", "frontend": "/static/"}


@app.get("/api/health")
async def health_check(brain: SmartAgentBrain = Depends(get_brain)):
    """健康检查端点"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "llm": brain.LLM_MODEL,
            "embedding": brain.EMBEDDING_MODEL,
            "cosyvoice": "loaded" if brain.local_model.cosyvoice_model else "not loaded",
            "funasr": "loaded" if brain.local_model.funasr_model else "not loaded",
            "cam": "loaded" if brain.local_model.CAM_model else "not loaded",
        },
        "speakers": list(speaker_manager.speakers.keys()) if speaker_manager else []
    }


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(default="Guest"),
    brain: SmartAgentBrain = Depends(get_brain)
):
    """文本对话接口（流式响应）"""
    return StreamingResponse(
        stream_chat_response(user_id, message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/asr")
async def asr_audio(
    audio: UploadFile = File(...),
    brain: SmartAgentBrain = Depends(get_brain)
):
    """音频识别接口"""
    # 保存上传的音频文件
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{datetime.now().timestamp()}.wav")

    try:
        # 保存上传的音频
        content = await audio.read()
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(content)

        # 执行 ASR
        res = await run_asr(temp_path)
        if res:
            raw_text = res[0]['text'].split(">")[-1].strip()
        else:
            raw_text = ""

        return {"success": True, "text": raw_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR 识别失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def audio_chat_generator(audio_file, user_id: str):
    """异步生成器：处理音频并流式返回响应"""
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{datetime.now().timestamp()}.wav")
    print(f"DEBUG: 正在尝试访问文件路径 -> {os.path.abspath(temp_path)}")
    try:
        # 1. 保存并识别音频
        content = await audio_file.read()
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(content)
        print(f"DEBUG: 文件已写入，大小: {os.path.getsize(temp_path)} bytes")
        res = await run_asr(temp_path)
        if res:
            user_text = res[0]['text'].split(">")[-1].strip()
        else:
            user_text = ""

        if not user_text:
            yield f"data: {json.dumps({'type': 'error', 'message': '未识别到有效语音'})}\n\n"
            return

        # 2. 发送识别结果
        yield f"data: {json.dumps({'type': 'asr_result', 'text': user_text})}\n\n"

        # 3. 流式处理对话
        full_text = ""
        async for chunk in agent_brain.process_user_query(user_text, user_id):
            if chunk.startswith("ACTION_REGISTER:"):
                yield f"data: {json.dumps({'type': 'action', 'action': 'register', 'target': chunk.split(':')[1]})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
            full_text += chunk

        # 4. 发送完成标记
        yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        # if os.path.exists(temp_path):
        #     os.remove(temp_path)
        pass


@app.post("/api/chat-audio")
async def chat_with_audio(
    audio: UploadFile = File(...),
    user_id: str = Form(default="Guest"),
    brain: SmartAgentBrain = Depends(get_brain)
):
    """音频转对话接口（一步到位，支持流式响应）"""
    return StreamingResponse(
        audio_chat_generator(audio, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/tts")
async def tts_text(
    text: str = Form(...),
    brain: SmartAgentBrain = Depends(get_brain)
):
    """文本转语音接口"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文本内容为空")

    try:
        audio_bytes = await run_tts(text)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{datetime.now().timestamp()}.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {str(e)}")


@app.post("/api/speaker/register")
async def register_speaker(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    spk_mgr: SpeakerManager = Depends(get_speaker_manager)
):
    """声纹注册接口"""
    if not user_id:
        raise HTTPException(status_code=400, detail="用户ID不能为空")

    # 保存音频文件
    filename = f"{user_id}.wav"
    save_path = os.path.join(SPEAKER_DIR, filename)

    try:
        content = await audio.read()
        async with aiofiles.open(save_path, "wb") as f:
            await f.write(content)

        # 刷新声纹库
        spk_mgr.refresh_speakers()

        return {
            "success": True,
            "user_id": user_id,
            "message": f"用户 {user_id} 声纹注册成功"
        }
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"声纹注册失败: {str(e)}")


@app.post("/api/speaker/identify")
async def identify_speaker(
    audio: UploadFile = File(...),
    spk_mgr: SpeakerManager = Depends(get_speaker_manager)
):
    """声纹识别接口"""
    temp_path = os.path.join(UPLOAD_DIR, f"temp_id_{datetime.now().timestamp()}.wav")

    try:
        # 保存音频
        content = await audio.read()
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(content)

        #识别
        result = spk_mgr.identify(temp_path)
        if isinstance(result, tuple) and len(result) == 2:
            user_id, score = result
        else:
            # 如果返回值格式不正确，使用默认值
            print(f"声纹识别返回值异常: {result}")
            user_id = "Unknown"
            score = 0.0

        return {
            "user_id": user_id,
            "score": float(score),
            "is_known": user_id != "Unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"声纹识别失败: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/speaker/list")
async def list_speakers(spk_mgr: SpeakerManager = Depends(get_speaker_manager)):
    """获取已注册用户列表"""
    return {
        "speakers": list(spk_mgr.speakers.keys()),
        "count": len(spk_mgr.speakers)
    }


@app.get("/api/memory/{user_id}")
async def get_user_memory(
    user_id: str,
    query: Optional[str] = None,
    brain: SmartAgentBrain = Depends(get_brain)
):
    """获取用户记忆"""
    try:
        if query:
            # 召回相关记忆
            memories = brain.recall_memories(query, user_id)
        else:
            # 获取图谱中的所有记忆
            memories = brain.kg.search_user_graph(user_id)

        return {
            "user_id": user_id,
            "memories": memories,
            "count": len(memories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记忆失败: {str(e)}")


# ==================== 主入口 ====================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
