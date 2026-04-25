import pyaudio
import wave
import threading
import numpy as np
import time
import queue  # 标准线程安全队列
import os
import asyncio
import pygame
import edge_tts
import glob
import re
import webrtcvad
import uuid
from pypinyin import pinyin, Style

# 本地模型和大脑
from SenseVoice_Agent_Brain import SmartAgentBrain
from SpeakerManager import SpeakerManager
from three_layer_pipeline import (
    InterfaceConfig,
    StateBridge,
    InterfaceLayer as DecoupledInterfaceLayer,
    SkillLayer as DecoupledSkillLayer,
    LLMLayer as DecoupledLLMLayer,
    ThreeLayerPipeline,
)
import torchaudio
import pyttsx3


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 可选: orchestrator | legacy | three-layer
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "orchestrator").strip().lower()
AGENT_SESSION_ID = os.environ.get("AGENT_SESSION_ID", f"agent-session-{uuid.uuid4().hex}")
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
VAD_MODE = 3
OUTPUT_DIR = "./output"
NO_SPEECH_THRESHOLD = 0.5  # 静音阈值从1->0.5s，适合更快的交互
folder_path = "./Test_Agent/"

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)


# 线程安全的队列，用于从录音线程传递音频路径到异步主循环
audio_task_queue = queue.Queue()

# 异步队列，用于 LLM 产出文本传给 TTS
tts_text_queue = asyncio.Queue()

recording_active = True
segments_to_save = []
last_active_time = time.time()
audio_file_count = 0

# 状态标志
is_speaking = False  # 正在播放音频
is_processing = False  # 正在进行 AI 推理

# --- KWS & 声纹配置 ---
set_KWS = "xiao ming tong xue"
flag_KWS = 0
flag_KWS_used = 1
flag_sv_used = 1
flag_sv_enroll = 0
thred_sv = 0.20
set_SV_enroll = r'.\SpeakerVerification_DIR\users\\'
temp_register_name = ""

# --- 初始化模型 ---
print("正在初始化模型，请稍候...")
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# 初始化大脑
agent_brain = SmartAgentBrain()
# 引用大脑里的模型实例
model_senceVoice = agent_brain.local_model.funasr_model
sv_pipeline = agent_brain.local_model.CAM_model

spk_manager = SpeakerManager(set_SV_enroll, sv_pipeline, threshold=0.15)
# 获取 CosyVoice 实例
# cosyvoice_model = agent_brain.local_model.cosyvoice_model

# --- 预生成/加载唤醒回复音频 ---
WAKEUP_FILE = os.path.join(folder_path, "wakeup_reply.mp3")

# 如果本地没有这个文件，就生成一个
if not os.path.exists(WAKEUP_FILE):
    print("正在预生成唤醒音频...")
    # 这里用同步方式生成一次即可，因为是在启动阶段
    async def gen_wakeup():
        communicate = edge_tts.Communicate("我在呢！", "zh-CN-XiaoyiNeural")
        await communicate.save(WAKEUP_FILE)
    asyncio.run(gen_wakeup())

# 全局初始化
pygame.mixer.init()

# print(">>> 模型加载完成！系统启动 (流式模式)！ <<<")
print(">>> 模型加载完成！系统启动 (全本地流式模式)！ <<<")

# --- 辅助函数 ---
def extract_pinyin(input_string):
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', input_string)
    chinese_text = ''.join(chinese_chars)
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    return ' '.join([item[0] for item in pinyin_result])


def play_audio_sync(file_path):
    """
    同步播放音频（会阻塞调用它的线程/协程），用于确保 TTS 句子按顺序说完
    """
    try:
        # pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # 降低CPU占用
        # pygame.mixer.quit()
    except Exception as e:
        print(f"播放失败: {e}")


async def async_play_audio(file_path):
    """异步包装播放，利用 executor 避免阻塞事件循环"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, play_audio_sync, file_path)


def start_turn_trace(audio_path: str, turn_id: str = ""):
    current_turn_id = turn_id or f"turn-{int(time.time() * 1000)}"

    # 每轮一个完整根 trace，避免长会话根 trace 导致langfuse平台可见性延迟。
    # 通过稳定 session_id 聚合同一会话的多轮 trace。
    return agent_brain.monitor.start_trace(
        name="agent_turn",
        user_id="pending",
        session_id=AGENT_SESSION_ID,
        input_payload={"audio_path": audio_path},
        metadata={
            "component": "SenseVoice_Agent_Main",
            "pipeline_mode": PIPELINE_MODE,
            "turn_id": current_turn_id,
        },
    )


def finalize_turn_trace(trace, output=None, metadata=None):
    if trace is None:
        return
    agent_brain.monitor.end_observation(trace, output=output, metadata=metadata)
    # turn_end 处优先保证可见性，采用同步 flush。
    agent_brain.monitor.flush_sync()


#  TTS 消费者

def pyttsx3_synthesis_sync(text, filename):
    """
    同步合成函数：每次调用都重新初始化 engine，避免 COM 冲突
    """
    import pyttsx3
    try:
        # 1. 每次都重新初始化
        engine = pyttsx3.init()

        # 2. 设置属性
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 10)  # 稍微调慢

        # 找中文发音人
        voices = engine.getProperty('voices')
        for v in voices:
            if "Chinese" in v.name or "Huihui" in v.name:
                engine.setProperty('voice', v.id)
                break

        # 3. 保存文件
        engine.save_to_file(text, filename)

        # 4. 执行并等待 (这是阻塞的)
        engine.runAndWait()

        # 5. 销毁引擎释放资源，避免内存泄漏和 COM 冲突
        del engine
        return True
    except Exception as e:
        print(f"pyttsx3 内部错误: {e}")
        return False


async def tts_consumer_worker():
    """
    后台任务：从队列取文本 -> 独立线程运行 pyttsx3 -> pygame 播放
    """
    global is_speaking, audio_file_count

    while True:
        # 等待队列中有文字
        item = await tts_text_queue.get()
        trace = None
        queued_at = None
        sentence_index = None

        if isinstance(item, dict):
            if item.get("type") == "turn_end":
                finalize_turn_trace(
                    item.get("trace"),
                    output=item.get("output"),
                    metadata=item.get("metadata"),
                )
                tts_text_queue.task_done()
                # turn_end 会提前 continue，需在此分支显式复位 speaking 状态。
                if tts_text_queue.empty():
                    is_speaking = False
                continue

            text = item.get("text", "")
            t0_ref = item.get("turn_t0")
            trace = item.get("trace")
            queued_at = item.get("queued_at")
            sentence_index = item.get("sentence_index")
        elif isinstance(item, tuple):
            if len(item) >= 5:
                text, t0_ref, trace, queued_at, sentence_index = item[:5]
            elif len(item) == 4:
                text, t0_ref, trace, queued_at = item
            elif len(item) == 3:
                text, t0_ref, trace = item
            elif len(item) == 2:
                text, t0_ref = item
            else:
                text, t0_ref = item, None
        else:
            text, t0_ref = item, None

        # 过滤无效文本
        if not text or len(text.strip()) < 1:
            tts_text_queue.task_done()
            continue

        is_speaking = True
        segments_to_save.clear()  # 防止录入自己声音

        try:
            audio_file_count += 1
            filename = os.path.join(folder_path, f"stream_{audio_file_count}.wav")

            print(f" pyttsx3 正在合成: {text}")

            tts_span = agent_brain.monitor.start_span(
                trace,
                name="tts_sentence",
                input_payload={"text": text, "sentence_index": sentence_index},
                metadata={"queued_at": queued_at, "turn_t0": t0_ref},
            )

            # 放入线程池运行
            loop = asyncio.get_event_loop()

            # run_in_executor 会在独立线程运行 pyttsx3_synthesis_sync, 这样 pyttsx3 的阻塞循环就不会卡死主程序的 asyncio 循环
            success = await loop.run_in_executor(None, pyttsx3_synthesis_sync, text, filename)

            if success and os.path.exists(filename):
                # 播放前打点
                t3_play = time.time()
                latency = t3_play - t0_ref if t0_ref else None
                queue_wait = t3_play - queued_at if queued_at else None
                if latency is not None:
                    print(f" T3--pyttsx3 响应延迟: {latency:.3f}s")

                # 播放
                await async_play_audio(filename)

                agent_brain.monitor.end_observation(
                    tts_span,
                    output={"filename": filename, "text": text},
                    metadata={
                        "latency_to_play_s": round(latency, 3) if latency is not None else None,
                        "queue_wait_s": round(queue_wait, 3) if queue_wait is not None else None,
                    },
                )

                # 删除临时文件
                try:
                    os.remove(filename)
                except:
                    pass
            else:
                print("TTS 合成失败或文件未生成")
                agent_brain.monitor.end_observation(
                    tts_span,
                    output={"filename": filename, "text": text},
                    metadata={"error": "tts_synthesis_failed"},
                )

        except Exception as e:
            print(f"TTS/播放出错: {e}")
            agent_brain.monitor.end_observation(
                tts_span if 'tts_span' in locals() else None,
                output={"text": text},
                metadata={"error": str(e)},
            )
        finally:
            tts_text_queue.task_done()
            if tts_text_queue.empty():
                is_speaking = False


class PipelineOrchestrator:
    """新编排器：封装 ASR/KWS/SV/LLM/TTS 主链路。"""

    async def run(self):
        global is_processing, flag_KWS, flag_sv_enroll, temp_register_name

        while True:
            turn_trace = None
            try:
                audio_path = audio_task_queue.get_nowait()
                t0_start = time.time()
                print(f"T0--开始处理: {t0_start}")
                turn_trace = start_turn_trace(audio_path, turn_id=f"orch-{int(t0_start * 1000)}")
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            is_processing = True
            segments_to_save.clear()

            try:
                existing_users = glob.glob(os.path.join(set_SV_enroll, "*.wav"))
                if flag_sv_used and not existing_users and not flag_sv_enroll:
                    print(" 初次接触，请注册。")
                    welcome_text = "欢迎使用，请说一句话注册声纹。"
                    await tts_text_queue.put({
                        "type": "tts_sentence",
                        "text": welcome_text,
                        "turn_t0": t0_start,
                        "trace": turn_trace,
                        "queued_at": time.time(),
                        "sentence_index": 0,
                    })
                    await tts_text_queue.put({
                        "type": "turn_end",
                        "trace": turn_trace,
                        "output": welcome_text,
                        "metadata": {"branch": "first_user_registration", "audio_path": audio_path},
                    })
                    temp_register_name = "主人"
                    flag_sv_enroll = 1
                    audio_task_queue.task_done()
                    is_processing = False
                    continue

                loop = asyncio.get_event_loop()
                raw_text = await loop.run_in_executor(None, run_asr, audio_path)
                t1_asr = time.time()
                print(f"T1--ASR耗时: {t1_asr - t0_start:.3f}s")
                agent_brain.monitor.update_observation(
                    turn_trace,
                    input_payload={"audio_path": audio_path, "asr_text": raw_text},
                    metadata={"asr_latency_s": round(t1_asr - t0_start, 3)},
                )
                if not raw_text:
                    finalize_turn_trace(turn_trace, output={"branch": "empty_asr"}, metadata={"audio_path": audio_path})
                    is_processing = False
                    continue

                pinyin_text = extract_pinyin(raw_text)
                print(f" 识别到: {raw_text}")

                if flag_KWS_used:
                    if set_KWS in pinyin_text:
                        print(">>> 唤醒成功！")
                        flag_KWS = 1
                        print("极速响应: 我在呢！")
                        await async_play_audio(WAKEUP_FILE)
                        finalize_turn_trace(
                            turn_trace,
                            output={"branch": "wakeup", "asr_text": raw_text},
                            metadata={"audio_path": audio_path, "user_id": "wakeword"},
                        )
                        is_processing = False
                        continue
                    if not flag_KWS:
                        finalize_turn_trace(
                            turn_trace,
                            output={"branch": "ignored_before_wakeup", "asr_text": raw_text},
                            metadata={"audio_path": audio_path},
                        )
                        is_processing = False
                        continue

                current_user_id = "Guest"
                if flag_sv_used:
                    user, score = spk_manager.identify(audio_path)
                    if user == "Unknown":
                        apology_text = "抱歉，我没听出你是谁。"
                        await tts_text_queue.put({
                            "type": "tts_sentence",
                            "text": apology_text,
                            "turn_t0": t0_start,
                            "trace": turn_trace,
                            "queued_at": time.time(),
                            "sentence_index": 0,
                        })
                        await tts_text_queue.put({
                            "type": "turn_end",
                            "trace": turn_trace,
                            "output": apology_text,
                            "metadata": {"audio_path": audio_path, "user_id": "Unknown"},
                        })
                        flag_KWS = 0
                        is_processing = False
                        continue
                    current_user_id = user
                    # 识别用户成功之后更新用户 ID 到 trace，方便后续分析
                    agent_brain.monitor.update_observation(turn_trace, metadata={"user_id": current_user_id, "audio_path": audio_path})

                first_sentence_flag = True
                sentence_index = 0
                full_response = ""
                turn_end_queued = False
                async for sentence in agent_brain.process_user_query(raw_text, current_user_id, trace=turn_trace):
                    if sentence.startswith("ACTION_REGISTER:"):
                        target_name = sentence.split(":")[1]
                        if target_name == "Unknown_User":
                            register_prompt = "请问怎么称呼您？"
                        else:
                            temp_register_name = target_name
                            flag_sv_enroll = 1
                            register_prompt = f"准备录入{target_name}的声纹，请听到滴声后说话。"
                        await tts_text_queue.put({
                            "type": "tts_sentence",
                            "text": register_prompt,
                            "turn_t0": t0_start,
                            "trace": turn_trace,
                            "queued_at": time.time(),
                            "sentence_index": sentence_index,
                        })
                        await tts_text_queue.put({
                            "type": "turn_end",
                            "trace": turn_trace,
                            "output": register_prompt,
                            "metadata": {"audio_path": audio_path, "user_id": current_user_id, "branch": "register_voice"},
                        })
                        turn_end_queued = True
                        break

                    if sentence.strip():
                        if first_sentence_flag:
                            t2_llm_first = time.time()
                            print(f"T2--LLM首句生成耗时: {t2_llm_first - t1_asr:.3f}s")
                            print(f"首句内容: {sentence}")
                        await tts_text_queue.put({
                            "type": "tts_sentence",
                            "text": sentence,
                            "turn_t0": t0_start,
                            "trace": turn_trace,
                            "queued_at": time.time(),
                            "sentence_index": sentence_index,
                        })
                        full_response += sentence
                        sentence_index += 1
                        first_sentence_flag = False

                if not turn_end_queued:
                    await tts_text_queue.put({
                        "type": "turn_end",
                        "trace": turn_trace,
                        "output": full_response,
                        "metadata": {"audio_path": audio_path, "user_id": current_user_id, "asr_text": raw_text},
                    })

            except Exception as e:
                print(f"Inference Error: {e}")
                finalize_turn_trace(turn_trace, output={"error": str(e)}, metadata={"audio_path": audio_path})
            finally:
                is_processing = False
                try:
                    os.remove(audio_path)
                except:
                    pass


async def inference_scheduler_legacy():
    """
    旧调度逻辑：保留用于灰度和回退。
    """
    global is_processing, flag_KWS, flag_sv_enroll, temp_register_name

    while True:
        # 1. 非阻塞方式检查 queue.Queue
        turn_trace = None
        try:
            # 从队列获取录音文件路径, 没取到就跳过
            audio_path = audio_task_queue.get_nowait()
            t0_start = time.time()  # <---【埋点 T0】开始处理
            print(f"T0--开始处理: {t0_start}")
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

        is_processing = True
        segments_to_save.clear()

        try:
            # --- 0. 首次运行注册逻辑 ---
            existing_users = glob.glob(os.path.join(set_SV_enroll, "*.wav"))  # 获取声纹库中所有已注册的 wav 文件路径，用于判断是否已有用户
            if flag_sv_used and not existing_users and not flag_sv_enroll:
                print("初次接触，请注册。")
                welcome_text = "欢迎使用，请说一句话注册声纹。"
                turn_trace = start_turn_trace(audio_path, turn_id=f"legacy-{int(t0_start * 1000)}")
                await tts_text_queue.put({
                    "type": "tts_sentence",
                    "text": welcome_text,
                    "turn_t0": t0_start,
                    "trace": turn_trace,
                    "queued_at": time.time(),
                    "sentence_index": 0,
                })
                await tts_text_queue.put({
                    "type": "turn_end",
                    "trace": turn_trace,
                    "output": welcome_text,
                    "metadata": {"branch": "first_user_registration", "audio_path": audio_path},
                })
                temp_register_name = "主人"
                flag_sv_enroll = 1
                audio_task_queue.task_done()
                is_processing = False
                continue

            # --- 1. ASR 识别 (同步模型需放入 executor 运行防止卡死) ---
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(None, run_asr, audio_path)
            t1_asr = time.time()  # <---【埋点 T1】ASR结束
            print(f"⏱️ [T1] ASR耗时: {t1_asr - t0_start:.3f}s")
            turn_trace = start_turn_trace(audio_path, turn_id=f"legacy-{int(t0_start * 1000)}")
            agent_brain.monitor.update_observation(
                turn_trace,
                input_payload={"audio_path": audio_path, "asr_text": raw_text},
                metadata={"asr_latency_s": round(t1_asr - t0_start, 3)},
            )
            if not raw_text:
                finalize_turn_trace(turn_trace, output={"branch": "empty_asr"}, metadata={"audio_path": audio_path})
                is_processing = False
                continue

            pinyin_text = extract_pinyin(raw_text)
            print(f"识别到: {raw_text}")

            # 2. 唤醒词逻辑 
            if flag_KWS_used:
                if set_KWS in pinyin_text:
                    print(">>> 唤醒成功！")
                    flag_KWS = 1
                    print("极速响应: 我在呢！")
                    await async_play_audio(WAKEUP_FILE)
                    finalize_turn_trace(
                        turn_trace,
                        output={"branch": "wakeup", "asr_text": raw_text},
                        metadata={"audio_path": audio_path, "user_id": "wakeword"},
                    )

                    # await tts_text_queue.put("我在呢！")  # 放入播放队列
                    is_processing = False  # 唤醒词不需要进LLM
                    continue
                else:
                    if not flag_KWS:
                        # 未唤醒状态，忽略
                        finalize_turn_trace(
                            turn_trace,
                            output={"branch": "ignored_before_wakeup", "asr_text": raw_text},
                            metadata={"audio_path": audio_path},
                        )
                        is_processing = False
                        continue

            # --- 3. 声纹识别 ---
            current_user_id = "Guest"
            if flag_sv_used:
                user, score = spk_manager.identify(audio_path)
                if user == "Unknown":
                    apology_text = "抱歉，我没听出你是谁。"
                    await tts_text_queue.put({
                        "type": "tts_sentence",
                        "text": apology_text,
                        "turn_t0": t0_start,
                        "trace": turn_trace,
                        "queued_at": time.time(),
                        "sentence_index": 0,
                    })
                    await tts_text_queue.put({
                        "type": "turn_end",
                        "trace": turn_trace,
                        "output": apology_text,
                        "metadata": {"audio_path": audio_path, "user_id": "Unknown"},
                    })
                    flag_KWS = 0
                    is_processing = False
                    continue
                current_user_id = user
                agent_brain.monitor.update_observation(turn_trace, metadata={"user_id": current_user_id, "audio_path": audio_path})

            # --- 4. LLM 流式交互 ---
            first_sentence_flag = True  # 标记是否是第一句
            sentence_index = 0
            full_response = ""
            turn_end_queued = False
            # 调用 Brain 的异步生成器
            async for sentence in agent_brain.process_user_query(raw_text, current_user_id, trace=turn_trace):

                # 检查特殊指令
                if sentence.startswith("ACTION_REGISTER:"):
                    target_name = sentence.split(":")[1]
                    if target_name == "Unknown_User":
                        register_prompt = "请问怎么称呼您？"
                    else:
                        temp_register_name = target_name
                        flag_sv_enroll = 1
                        register_prompt = f"准备录入{target_name}的声纹，请听到滴声后说话。"
                    await tts_text_queue.put({
                        "type": "tts_sentence",
                        "text": register_prompt,
                        "turn_t0": t0_start,
                        "trace": turn_trace,
                        "queued_at": time.time(),
                        "sentence_index": sentence_index,
                    })
                    await tts_text_queue.put({
                        "type": "turn_end",
                        "trace": turn_trace,
                        "output": register_prompt,
                        "metadata": {"audio_path": audio_path, "user_id": current_user_id, "branch": "register_voice"},
                    })
                    turn_end_queued = True
                    break  # 停止后续生成

                # 普通文本 -> 放入播放队列
                if sentence.strip():
                    if first_sentence_flag:
                        t2_llm_first = time.time()  # <---【埋点 T2】LLM生成首句
                        print(f"T2--LLM首句生成耗时: {t2_llm_first - t1_asr:.3f}s")
                        print(f"首句内容: {sentence}")

                    await tts_text_queue.put({
                        "type": "tts_sentence",
                        "text": sentence,
                        "turn_t0": t0_start,
                        "trace": turn_trace,
                        "queued_at": time.time(),
                        "sentence_index": sentence_index,
                    })
                    full_response += sentence
                    sentence_index += 1
                    first_sentence_flag = False

            if not turn_end_queued:
                await tts_text_queue.put({
                    "type": "turn_end",
                    "trace": turn_trace,
                    "output": full_response,
                    "metadata": {"audio_path": audio_path, "user_id": current_user_id, "asr_text": raw_text},
                })

        except Exception as e:
            print(f"Inference Error: {e}")
            finalize_turn_trace(turn_trace, output={"error": str(e)}, metadata={"audio_path": audio_path})
        finally:
            is_processing = False
            # 删除录音临时文件
            try:
                os.remove(audio_path)
            except:
                pass


async def inference_scheduler():
    """统一调度入口：按模式选择 orchestrator / legacy / three-layer。"""
    mode = PIPELINE_MODE
    if mode not in ("legacy", "orchestrator", "three-layer"):
        print(f"[Scheduler] 未知 PIPELINE_MODE={mode}，自动回退 orchestrator")
        mode = "orchestrator"

    if mode == "legacy":
        print("[Scheduler] 当前模式: legacy")
        await inference_scheduler_legacy()
        return

    if mode == "three-layer":
        print("[Scheduler] 当前模式: three-layer")

        def set_processing(v: bool):
            global is_processing
            is_processing = v

        def clear_segments():
            segments_to_save.clear()

        def get_flag_kws() -> int:
            return flag_KWS

        def set_flag_kws(v: int):
            global flag_KWS
            flag_KWS = v

        def get_flag_sv_enroll() -> int:
            return flag_sv_enroll

        def set_flag_sv_enroll(v: int):
            global flag_sv_enroll
            flag_sv_enroll = v

        def set_temp_register_name(v: str):
            global temp_register_name
            temp_register_name = v

        state_bridge = StateBridge(
            set_processing=set_processing,
            clear_segments=clear_segments,
            get_flag_kws=get_flag_kws,
            set_flag_kws=set_flag_kws,
            get_flag_sv_enroll=get_flag_sv_enroll,
            set_flag_sv_enroll=set_flag_sv_enroll,
            set_temp_register_name=set_temp_register_name,
        )

        interface_config = InterfaceConfig(
            kws_text=set_KWS,
            kws_enabled=bool(flag_KWS_used),
            sv_enabled=bool(flag_sv_used),
            enroll_dir=set_SV_enroll,
        )

        interface_layer = DecoupledInterfaceLayer(
            audio_task_queue=audio_task_queue,
            tts_text_queue=tts_text_queue,
            asr_func=run_asr,
            extract_pinyin_func=extract_pinyin,
            speaker_identify_func=spk_manager.identify,
            play_wakeup_func=async_play_audio,
            wakeup_file=WAKEUP_FILE,
            config=interface_config,
            state=state_bridge,
        )
        skill_layer = DecoupledSkillLayer(
            intent_router_service=agent_brain.intent_router_service,
            skill_orchestrator=agent_brain.skill_orchestrator,
        )
        llm_layer = DecoupledLLMLayer(agent_brain)
        pipeline = ThreeLayerPipeline(
            interface_layer=interface_layer,
            skill_layer=skill_layer,
            llm_layer=llm_layer,
            state=state_bridge,
        )
        await pipeline.run()
        return

    print("[Scheduler] 当前模式: orchestrator")
    orchestrator = PipelineOrchestrator()
    await orchestrator.run()


def run_asr(audio_path):
    """封装 ASR 为独立函数"""
    try:
        res = model_senceVoice.generate(input=audio_path, cache={}, language="auto", use_itn=False)
        return res[0]['text'].split(">")[-1].strip()
    except:
        return ""


# --- 录音线程 (保持独立) ---
def audio_recorder_thread():
    global recording_active, last_active_time, segments_to_save, audio_file_count, flag_sv_enroll, temp_register_name

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                    frames_per_buffer=CHUNK)
    audio_buffer = []

    print("麦克风监听中...")

    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

        # VAD 检测逻辑 (每0.5秒)
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            is_speech = is_speech_detected(raw_audio)

            if is_speech:
                # 只有当机器人没在说话、也没在思考时，才录音
                if not is_speaking and not is_processing:
                    last_active_time = time.time()
                    segments_to_save.append((raw_audio, time.time()))
                else:
                    # 如果机器人在说话，清空buffer防止录入回声
                    pass

            audio_buffer = []  # 重置 buffer

        # 判定句子结束
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD and segments_to_save:
            # 再次检查，防止截断播报
            if is_speaking or is_processing:
                segments_to_save.clear()
                continue

            # 保存逻辑
            if flag_sv_enroll:
                handle_enrollment()
            else:
                # 普通对话 -> 保存临时文件 -> 放入队列
                save_temp_and_queue()

    stream.stop_stream()
    stream.close()
    p.terminate()


def is_speech_detected(raw_audio):
    """VAD 检测封装"""
    step = int(AUDIO_RATE * 0.02)
    frames = 0
    for i in range(0, len(raw_audio), step):
        chunk = raw_audio[i:i + step]
        if len(chunk) == step and vad.is_speech(chunk, AUDIO_RATE):
            frames += 1
    return frames > 5


def save_temp_and_queue():
    """保存对话录音并放入处理队列"""
    global segments_to_save
    temp_path = f"{OUTPUT_DIR}/rec_{int(time.time())}.wav"
    write_wav(temp_path, segments_to_save)
    segments_to_save.clear()

    # 放入队列，通知主线程处理
    audio_task_queue.put(temp_path)


def handle_enrollment():
    """处理声纹注册逻辑"""
    global flag_sv_enroll, temp_register_name, segments_to_save

    final_name = f"{temp_register_name}.wav" if temp_register_name else f"User_{int(time.time())}.wav"
    save_path = os.path.join(set_SV_enroll, final_name)

    write_wav(save_path, segments_to_save)
    segments_to_save.clear()

    print(f"声纹已注册: {final_name}")
    spk_manager.refresh_speakers()
    flag_sv_enroll = 0
    temp_register_name = ""

    # 这里的反馈建议直接打印，或者放入TTS队列
    # 由于我们在子线程，不能直接调 async 函数，使用 run_coroutine_threadsafe 或者简单地忽略
    print("注册完成，请继续对话。")


def write_wav(path, segments):
    wf = wave.open(path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join([seg[0] for seg in segments]))
    wf.close()


# 主入口
async def main_entry():
    # 启动 TTS 消费者任务
    tts_task = asyncio.create_task(tts_consumer_worker())

    # 启动 推理调度任务
    inference_task = asyncio.create_task(inference_scheduler())

    # 启动 录音线程
    rec_thread = threading.Thread(target=audio_recorder_thread, daemon=True)
    rec_thread.start()

    print("所有服务已就绪，请说话...")

    # 等待任务 (实际上是无限循环)
    await asyncio.gather(tts_task, inference_task)


if __name__ == "__main__":
    try:
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        recording_active = False
        print("系统退出")
    finally:
        try:
            agent_brain.monitor.flush_sync()
        except Exception:
            pass