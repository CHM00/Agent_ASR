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
from pypinyin import pinyin, Style

from SenseVoice_Agent_Brain import SmartAgentBrain
from SpeakerManager import SpeakerManager
import torchaudio
import pyttsx3

# --- 配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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

# --- 全局状态 ---
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
thred_sv = 0.30
set_SV_enroll = r'.\SpeakerVerification_DIR\users\\'
temp_register_name = ""

# --- 初始化模型 ---
print("正在初始化模型，请稍候...")
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# 初始化大脑
agent_brain = SmartAgentBrain()

model_senceVoice = agent_brain.local_model.funasr_model
sv_pipeline = agent_brain.local_model.CAM_model

spk_manager = SpeakerManager(set_SV_enroll, sv_pipeline, threshold=0.35)
# 获取 CosyVoice 实例
cosyvoice_model = agent_brain.local_model.cosyvoice_model

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

# 全局初始化 pygame mixer
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


# --- TTS 消费者（EdgeTTS, CosyVoice, pyttsx3） ---
def pyttsx3_synthesis_sync(text, filename):
    """
    同步合成函数：每次调用都重新初始化 engine，避免 COM 冲突
    """
    import pyttsx3
    try:
        # 1. 每次都重新初始化 (关键！)
        engine = pyttsx3.init()

        # 2. 设置属性 (语速、音色)
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 10)  # 稍微调慢

        # 尝试找中文发音人
        voices = engine.getProperty('voices')
        for v in voices:
            if "Chinese" in v.name or "Huihui" in v.name:
                engine.setProperty('voice', v.id)
                break

        # 3. 保存文件
        # 注意：save_to_file 是将命令放入队列
        engine.save_to_file(text, filename)

        # 4. 执行并等待 (这是阻塞的)
        engine.runAndWait()

        # 5. 销毁引擎 (虽然 Python 会自动回收，但在 COM 中显式清理更好)
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
        if isinstance(item, tuple):
            text, t0_ref = item
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

            print(f"pyttsx3 正在合成: {text}")

            # 放入线程池运行
            loop = asyncio.get_event_loop()

            # run_in_executor 会在独立线程运行 pyttsx3_synthesis_sync
            # 这样 pyttsx3 的阻塞循环就不会卡死主程序的 asyncio 循环
            success = await loop.run_in_executor(None, pyttsx3_synthesis_sync, text, filename)

            if success and os.path.exists(filename):
                # 播放前打点
                t3_play = time.time()
                if t0_ref:
                    latency = t3_play - t0_ref
                    print(f"T3-pyttsx3 响应延迟: {latency:.3f}s")

                # 播放
                await async_play_audio(filename)

                # 删除临时文件
                try:
                    os.remove(filename)
                except:
                    pass
            else:
                print("TTS 合成失败或文件未生成")

        except Exception as e:
            print(f"TTS/播放出错: {e}")
        finally:
            tts_text_queue.task_done()
            if tts_text_queue.empty():
                is_speaking = False

'''
    使用CosyVoice 合成
'''
async def tts_consumer_worker():
    """
    后台任务：不断从 tts_text_queue 获取文字 -> CosyVoice 合成 -> 播放
    """
    global is_speaking, audio_file_count

    while True:
        item = await tts_text_queue.get()
        if isinstance(item, tuple):
            text, t0_ref = item
        else:
            text, t0_ref = item, None

        # 过滤掉空的或者太短的文本，避免报错
        if not text or len(text.strip()) < 1:
            tts_text_queue.task_done()
            continue

        is_speaking = True
        segments_to_save.clear()

        try:
            audio_file_count += 1
            # CosyVoice 输出是 wav 格式，建议用 .wav 后缀
            filename = os.path.join(folder_path, f"stream_{audio_file_count}.wav")

            print(f"正在合成: {text}")

            # --- CosyVoice 推理 (这是同步代码，且耗时，必须放入线程池) ---
            loop = asyncio.get_event_loop()

            # 定义一个同步函数来执行推理和保存
            def run_cosyvoice_sync(input_text, out_path):
                # stream=False: 咱们是按句子进来的，直接生成整句比流式切片处理简单且效果好
                # '中文女' 是音色名，你可以改成 '中文男' 或其他
                model_output = cosyvoice_model.inference_sft(input_text, '中文女', stream=False)

                # 遍历生成器 (其实只有一段音频，因为 stream=False)
                for i, j in enumerate(model_output):
                    # j['tts_speech'] 是 tensor, 采样率通常是 22050
                    torchaudio.save(out_path, j['tts_speech'], 22050)
                    return True  # 生成成功
                return False

            # 在 executor 中运行，不阻塞主线程
            success = await loop.run_in_executor(None, run_cosyvoice_sync, text, filename)

            if success:
                # 播放前打点
                t3_play = time.time()
                if t0_ref:
                    latency = t3_play - t0_ref
                    print(f"T3-CosyVoice 响应延迟: {latency:.3f}s")

                # 播放
                await async_play_audio(filename)

                # 删除
                try:
                    os.remove(filename)
                except:
                    pass
            else:
                print("CosyVoice 合成未返回数据")

        except Exception as e:
            print(f"TTS/播放出错: {e}")
        finally:
            tts_text_queue.task_done()
            if tts_text_queue.empty():
                is_speaking = False

'''
使用 EdgeTTS 合成
'''
async def tts_consumer_worker():
    """
    后台任务：不断从 tts_text_queue 获取文字 -> 合成 -> 播放
    实现了“边生成边播放”的效果，且保证句子顺序
    """
    global is_speaking, audio_file_count

    while True:
        # 等待队列中有文字
        item = await tts_text_queue.get()
        if isinstance(item, tuple):
            text, t0_ref = item
        else:
            text, t0_ref = item, None

        # text = await tts_text_queue.get()

        # 标记正在说话
        is_speaking = True
        # 清空录音缓存，防止把自己说话录进去
        segments_to_save.clear()

        try:
            audio_file_count += 1
            filename = os.path.join(folder_path, f"stream_{audio_file_count}.mp3")

            print(f"正在播报: {text}")

            # 1. 生成语音 Edge-TTS 是异步的
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural")
            await communicate.save(filename)

            # 播放前打点
            t3_play = time.time()  # <---【埋点 T3】开始播放
            if t0_ref:
                latency = t3_play - t0_ref
                print(f"T3-首字音频响应延迟！！！: {latency:.3f}s")

            # 2. 播放语音 (必须await播放完成，否则下一句会重叠)
            await async_play_audio(filename)

            # 3. 删除临时文件
            try:
                os.remove(filename)
            except:
                pass

        except Exception as e:
            print(f"TTS/播放出错: {e}")
        finally:
            tts_text_queue.task_done()

            # 如果队列空了，说明这一轮说话结束
            if tts_text_queue.empty():
                is_speaking = False


# --- 核心任务：推理调度器 ---
async def inference_scheduler():
    """
    主循环：监听录音线程发来的音频路径 -> 执行 ASR -> 执行 LLM -> 推送给 TTS
    """
    global is_processing, flag_KWS, flag_sv_enroll, temp_register_name

    while True:
        # 1. 非阻塞方式检查 queue.Queue
        try:
            # 从队列获取录音文件路径, 没取到就跳过
            audio_path = audio_task_queue.get_nowait()
            t0_start = time.time()  # <---【埋点 T0】开始处理
            print(f"T0-开始处理: {t0_start}")
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

        is_processing = True
        segments_to_save.clear()

        try:
            # --- 0. 首次运行注册逻辑 ---
            existing_users = glob.glob(os.path.join(set_SV_enroll, "*.wav"))
            if flag_sv_used and not existing_users and not flag_sv_enroll:
                print("初次见面，请注册。")
                await tts_text_queue.put("欢迎使用，请说一句话注册声纹。")
                temp_register_name = "主人"
                flag_sv_enroll = 1
                audio_task_queue.task_done()
                is_processing = False
                continue

            # --- 1. ASR 识别 (同步模型需放入 executor 运行防止卡死) ---
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(None, run_asr, audio_path)
            t1_asr = time.time()  # <---【埋点 T1】ASR结束
            print(f"T1-ASR耗时: {t1_asr - t0_start:.3f}s")
            if not raw_text:
                is_processing = False
                continue

            pinyin_text = extract_pinyin(raw_text)
            print(f"听到: {raw_text}")

            # --- 2. 唤醒词逻辑 ---
            if flag_KWS_used:
                if set_KWS in pinyin_text:
                    print(">>> 唤醒成功！")
                    flag_KWS = 1
                    print("极速响应: 我在呢！")
                    await async_play_audio(WAKEUP_FILE)

                    # await tts_text_queue.put("我在呢！")  # 放入播放队列
                    is_processing = False  # 唤醒词不需要进LLM
                    continue
                else:
                    if not flag_KWS:
                        # 未唤醒状态，忽略
                        is_processing = False
                        continue

            # --- 3. 声纹识别 ---
            current_user_id = "Guest"
            if flag_sv_used:
                user, score = spk_manager.identify(audio_path)
                if user == "Unknown":
                    await tts_text_queue.put("抱歉，我没听出你是谁。")
                    flag_KWS = 0
                    is_processing = False
                    continue
                current_user_id = user

            # --- 4. LLM 流式交互 ---
            first_sentence_flag = True  # 标记是否是第一句
            # 调用 Brain 的异步生成器
            async for sentence in agent_brain.process_user_query(raw_text, current_user_id):

                # 检查特殊指令
                if sentence.startswith("ACTION_REGISTER:"):
                    target_name = sentence.split(":")[1]
                    if target_name == "Unknown_User":
                        await tts_text_queue.put("请问怎么称呼您？")
                    else:
                        temp_register_name = target_name
                        flag_sv_enroll = 1
                        await tts_text_queue.put(f"准备录入{target_name}的声纹，请听到滴声后说话。")
                    break  # 停止后续生成

                # 普通文本 -> 放入播放队列
                if sentence.strip():
                    if first_sentence_flag:
                        t2_llm_first = time.time()
                        print(f"T2-LLM首句生成耗时: {t2_llm_first - t1_asr:.3f}s")
                        print(f"首句内容: {sentence}")
                        # 将 t0 传给 TTS 队列以便计算总延迟

                    await tts_text_queue.put((sentence, t0_start if first_sentence_flag else None))
                    first_sentence_flag = False
                    # await tts_text_queue.put(sentence)

        except Exception as e:
            print(f"Inference Error: {e}")
        finally:
            is_processing = False
            # 删除录音临时文件
            try:
                os.remove(audio_path)
            except:
                pass


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

    print("🎙️ 麦克风监听中...")

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

            audio_buffer = []  # 重置buffer

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

    # 由于我们在子线程，不能直接调 async 函数，使用 run_coroutine_threadsafe
    print("注册完成，请继续对话。")


def write_wav(path, segments):
    wf = wave.open(path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join([seg[0] for seg in segments]))
    wf.close()


# --- 主入口 ---
async def main_entry():
    # 启动 TTS 消费者任务
    tts_task = asyncio.create_task(tts_consumer_worker())

    # 启动 推理调度任务
    inference_task = asyncio.create_task(inference_scheduler())

    # 启动 录音线程
    rec_thread = threading.Thread(target=audio_recorder_thread, daemon=True)
    rec_thread.start()

    print("所有服务已就绪，请说话...")

    # 等待任务
    await asyncio.gather(tts_task, inference_task)


if __name__ == "__main__":
    try:
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        recording_active = False
        print("系统退出")