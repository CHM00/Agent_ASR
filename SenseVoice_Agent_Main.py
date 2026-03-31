# # filename: agent_main.py
# import cv2
# import pyaudio
# import wave
# import threading
# import numpy as np
# import time
# from queue import Queue
# import webrtcvad
# import os
# import asyncio
# import pygame
# import edge_tts
# from funasr import AutoModel
# from modelscope.pipelines import pipeline
# from pypinyin import pinyin, Style
# import re
# from SpeakerManager import SpeakerManager
# # --- 导入我们的大脑 ---
# from SenseVoice_Agent_Brain_Copy_New import SmartAgentBrain
# import glob
# import asyncio
# # --- 配置 ---
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# AUDIO_RATE = 16000
# AUDIO_CHANNELS = 1
# CHUNK = 1024
# VAD_MODE = 3
# OUTPUT_DIR = "./output"
# NO_SPEECH_THRESHOLD = 1
# folder_path = "./Test_Agent/"
#
# # 确保目录存在
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(folder_path, exist_ok=True)
#
# # 全局变量
# audio_queue = Queue()
# recording_active = True
# segments_to_save = []
# last_active_time = time.time()
# last_vad_end_time = 0
# audio_file_count = 0
#
# # --- KWS & 声纹配置 ---
# set_KWS = "xiao ming tong xue"  # 唤醒词拼音
# flag_KWS = 0
# flag_KWS_used = 1  # 是否开启唤醒词
# flag_sv_used = 1  # 是否开启声纹
# flag_sv_enroll = 0  # 是否处于注册模式
# thred_sv = 0.30  # 声纹阈值
#
# is_speaking = False  # 是否正在播放语音
# is_processing = False  # 是否正在处理推理（ASR+LLM+TTS整个流程）
#
# # 声纹路径
# # set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
# set_SV_enroll = r'.\SpeakerVerification_DIR\users\\'
# temp_register_name = "" # 用于暂存即将注册的用户名
# # --- 初始化模型 ---
# print("正在初始化模型，请稍候...")
#
# # 初始化 VAD
# vad = webrtcvad.Vad()
# vad.set_mode(VAD_MODE)
#
#
# # 初始化 Agent 大脑 (连接 Milvus 和 LLM)
# agent_brain = SmartAgentBrain()
# model_senceVoice = agent_brain.local_model.funasr_model
# sv_pipeline = agent_brain.local_model.CAM_model
#
# # 初始化多用户管理器
# spk_manager = SpeakerManager(set_SV_enroll, agent_brain.local_model.CAM_model, threshold=0.35)
# print(">>> 模型加载完成！系统启动！ <<<")
#
# speech_queue = asyncio.Queue()
#
#
# async def tts_worker():
#     """专门负责从队列取文本 -> 合成 -> 播放的后台任务"""
#     global is_speaking, audio_file_count
#     while True:
#         text = await speech_queue.get()
#         if not text:
#             speech_queue.task_done()
#             continue
#
#         is_speaking = True
#         audio_file_count += 1
#         filename = os.path.join(folder_path, f"stream_{audio_file_count}.mp3")
#
#         try:
#             # 1. 合成
#             communicate = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural")
#             await communicate.save(filename)
#
#             # 2. 播放 (在 executor 中运行 pygame 避免阻塞)
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(None, play_audio, filename)
#
#             # 3. 播放完后删除临时文件
#             if os.path.exists(filename):
#                 os.remove(filename)
#         except Exception as e:
#             print(f"流式播放失败: {e}")
#         finally:
#             is_speaking = False
#             speech_queue.task_done()
#
#
# # --- 辅助函数 ---
# def extract_pinyin(input_string):
#     chinese_chars = re.findall(r'[\u4e00-\u9fa5]', input_string)
#     chinese_text = ''.join(chinese_chars)
#     pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
#     return ' '.join([item[0] for item in pinyin_result])
#
#
# def play_audio(file_path):
#     try:
#         pygame.mixer.init()
#         pygame.mixer.music.load(file_path)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             time.sleep(0.1)
#         pygame.mixer.quit()
#     except Exception as e:
#         print(f"播放失败: {e}")
#
#
# async def text_to_speech(text, output_file):
#     """使用 Edge TTS 生成语音"""
#     voice = "zh-CN-XiaoyiNeural"
#     communicate = edge_tts.Communicate(text, voice)
#     await communicate.save(output_file)
#
#
# def system_speak(text):
#     """同步包装的 TTS 播放函数"""
#     global audio_file_count, is_speaking, segments_to_save
#
#     is_speaking = True
#     segments_to_save.clear() # 清空之前的缓存，避免录入播报声音
#     print(f"Agent Output: {text}")
#     audio_file_count += 1
#     filename = os.path.join(folder_path, f"reply_{audio_file_count}.mp3")
#     asyncio.run(text_to_speech(text, filename))
#     play_audio(filename)
#
#     time.sleep(0.3)
#     is_speaking = False
#
#
# # --- 核心推理线程 ---
# def Inference(audio_path):
#     global flag_sv_enroll, flag_KWS, flag_KWS_used, flag_sv_used, set_SV_enroll, is_processing, segments_to_save, temp_register_name
#
#     is_processing = True  # 开始处理，暂停录音
#     segments_to_save.clear()
#     current_user_id = "Guest"  # 默认为访客
#     try:
#         # 0. 检查声纹文件夹是否为空 (初次运行逻辑)
#         existing_users = glob.glob(os.path.join(set_SV_enroll, "*.wav"))
#         if flag_sv_used and not existing_users:
#             print("声纹库为空，进入首个用户注册模式...")
#             system_speak("欢迎使用，我需要先认识你。请说一句话大于3s的句子用于注册声纹。")
#             temp_register_name = "主人"  # 默认第一个人叫主人
#             flag_sv_enroll = 1
#             return
#
#         # 1. ASR 语音识别
#         try:
#             res = model_senceVoice.generate(input=audio_path, cache={}, language="auto", use_itn=False)
#             raw_text = res[0]['text'].split(">")[-1].strip()
#             pinyin_text = extract_pinyin(raw_text)
#             print(f"听到: {raw_text} (拼音: {pinyin_text})")
#         except Exception as e:
#             print(f"ASR Error: {e}")
#             return
#
#         if not raw_text: return
#
#         # 2. 唤醒词检测 (KWS)
#         if flag_KWS_used:
#             if set_KWS in pinyin_text:
#                 print(">>> 唤醒词匹配成功！")
#                 flag_KWS = 1
#                 # 唤醒成功, 播报
#                 system_speak("你好, 我在呢!")
#                 return
#             else:
#                 # 如果没唤醒，直接忽略
#                 if not flag_KWS:
#                     print("未唤醒...")
#                     return
#         # 2. 声纹对比 (CAM)
#         if flag_sv_used:
#             try:
#                 identified_user, score = spk_manager.identify(audio_path)
#
#                 if identified_user == "Unknown":
#                     system_speak("身份验证失败，我不认识你。")
#                     flag_KWS = 0
#                     return  # 拒绝执行
#
#                 current_user_id = identified_user
#                 print(f"识别成功，当前用户: {current_user_id}")
#
#             except Exception as e:
#                 print(f"SV Error: {e}")
#                 return
#
#         # 4. 调用 Agent 处理
#         # 使用 asyncio.run 在同步线程中调用异步逻辑
#         reply = asyncio.run(agent_brain.process_user_query(raw_text, user_id=current_user_id))
#
#         # 检查是否是注册指令
#         if reply.startswith("ACTION_REGISTER:"):
#             target_name = reply.split(":")[1]
#             if target_name == "Unknown_User":
#                 system_speak("好的，请告诉我你怎么称呼？")
#             else:
#                 temp_register_name = target_name
#                 flag_sv_enroll = 1  # 开启注册模式
#                 system_speak(f"好的，准备录入【{target_name}】的声纹。请在听到‘滴’声后，清晰地说一句话，至少3秒。")
#                 # 说完后, 进入下一轮 audio_recorder
#             return
#
#         # 5. 播报结果
#         system_speak(reply)
#         pass
#     finally:
#         is_processing = False  # 处理完成，恢复录音
#
#     # 交互完成后，可以选择重置唤醒状态 (需再次唤醒)，或者保持唤醒
#     # flag_KWS = 0
#
#
# # --- 录音线程 ---
# def audio_recorder():
#     global recording_active, last_active_time, segments_to_save, last_vad_end_time, audio_file_count, flag_sv_enroll, temp_register_name
#
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
#                     frames_per_buffer=CHUNK)
#     audio_buffer = []
#
#     print("麦克风监听中...")
#
#     while recording_active:
#         data = stream.read(CHUNK)
#         audio_buffer.append(data)
#
#         # 每 0.5 秒检测 VAD
#         if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
#             raw_audio = b''.join(audio_buffer)
#             # 简单 VAD 检测
#             is_speech = False
#             step = int(AUDIO_RATE * 0.02)
#             speech_frames = 0
#             for i in range(0, len(raw_audio), step):
#                 chunk = raw_audio[i:i + step]
#                 if len(chunk) == step and vad.is_speech(chunk, AUDIO_RATE):
#                     speech_frames += 1
#             if speech_frames > 5: is_speech = True
#
#             if is_speech:
#                 # 正在处理或播报时不记录语音段
#                 if not is_speaking and not is_processing:
#                     last_active_time = time.time()
#                     segments_to_save.append((raw_audio, time.time()))
#
#             audio_buffer = []
#
#         # 判定句子结束 (静音超时)
#         if time.time() - last_active_time > NO_SPEECH_THRESHOLD and segments_to_save:
#
#             # 正在播报，跳过处理并清空缓存
#             if is_speaking or is_processing:
#                 segments_to_save.clear()
#                 continue
#             # 保存并推理
#             audio_file_count += 1
#
#             # 处理声纹注册的特殊逻辑
#             save_path = f"{OUTPUT_DIR}/audio_tmp.wav"
#             if flag_sv_enroll:
#                 os.makedirs(set_SV_enroll, exist_ok=True)
#                 # 如果没有名字，就用时间戳兜底
#                 if not temp_register_name:
#                     final_name = f"User_{int(time.time())}.wav"
#                 else:
#                     final_name = f"{temp_register_name}.wav"
#
#                 save_path = os.path.join(set_SV_enroll, final_name)
#
#             # 写入文件
#             wf = wave.open(save_path, 'wb')
#             wf.setnchannels(AUDIO_CHANNELS)
#             wf.setsampwidth(2)
#             wf.setframerate(AUDIO_RATE)
#             wf.writeframes(b''.join([seg[0] for seg in segments_to_save]))
#             wf.close()
#
#             segments_to_save.clear()  # 清空缓存
#
#             if flag_sv_enroll:
#                 print(f"声纹注册文件已保存: {save_path}")
#                 flag_sv_enroll = 0  # 关闭开关
#                 temp_register_name = ""  # 清空暂存名
#                 # SpeakerManager 刷新用户列表
#                 try:
#                     spk_manager.refresh_speakers()
#                 except:
#                     pass
#
#                 system_speak("注册成功！我已经记住你的声音了。")
#             else:
#                 # 开启新线程推理，避免阻塞录音
#                 t = threading.Thread(target=Inference, args=(save_path,))
#                 t.start()
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#
# # if __name__ == "__main__":
# #     try:
# #         t_rec = threading.Thread(target=audio_recorder)
# #         t_rec.start()
# #
# #         while True:
# #             time.sleep(1)
# #     except KeyboardInterrupt:
# #         recording_active = False
#
# if __name__ == "__main__":
#     try:
#         # 启动录音线程
#         t_rec = threading.Thread(target=audio_recorder, daemon=True)
#         t_rec.start()
#
#
#         # 启动异步事件循环来运行 TTS Worker
#         async def main_async():
#             # 启动播放消费者任务
#             worker_task = asyncio.create_task(tts_worker())
#             print(">>> 系统已进入流式响应模式 <<<")
#             while True:
#                 await asyncio.sleep(1)
#
#
#         asyncio.run(main_async())
#
#     except KeyboardInterrupt:
#         recording_active = False


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

# --- 导入我们的大脑 ---
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


# --- 核心任务：TTS 消费者 ---
# ================= 核心修改：TTS 消费者 Worker =================

def pyttsx3_synthesis_sync(text, filename):
    """
    同步合成函数：每次调用都重新初始化 engine，避免 COM 冲突
    """
    import pyttsx3
    try:
        # 1. 每次都重新初始化
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

            print(f" pyttsx3 正在合成: {text}")

            # --- 关键修改：放入线程池运行 ---
            loop = asyncio.get_event_loop()

            # run_in_executor 会在独立线程运行 pyttsx3_synthesis_sync
            # 这样 pyttsx3 的阻塞循环就不会卡死主程序的 asyncio 循环
            success = await loop.run_in_executor(None, pyttsx3_synthesis_sync, text, filename)

            if success and os.path.exists(filename):
                # 播放前打点
                t3_play = time.time()
                if t0_ref:
                    latency = t3_play - t0_ref
                    print(f"🚀 [T3] pyttsx3 响应延迟: {latency:.3f}s")

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


# async def tts_consumer_worker():
#     """
#     后台任务：从队列取文本 -> pyttsx3 合成文件 -> pygame 播放
#     """
#     global is_speaking, audio_file_count
#
#     # 1. 初始化 pyttsx3 引擎
#     engine = pyttsx3.init()
#
#     # 设置语速 (可选，默认通常是 200)
#     rate = engine.getProperty('rate')
#     engine.setProperty('rate', rate - 20)  # 稍微慢一点点，更清晰
#
#     # 设置中文语音 (尝试自动找中文语音包)
#     voices = engine.getProperty('voices')
#     for voice in voices:
#         # Windows 上通常包含 'Chinese' 或 'Huihui'
#         if "Chinese" in voice.name or "Huihui" in voice.name:
#             engine.setProperty('voice', voice.id)
#             break
#
#     while True:
#         # 等待队列中有文字
#         item = await tts_text_queue.get()
#         if isinstance(item, tuple):
#             text, t0_ref = item
#         else:
#             text, t0_ref = item, None
#
#         if not text or len(text.strip()) < 1:
#             tts_text_queue.task_done()
#             continue
#
#         is_speaking = True
#         segments_to_save.clear()  # 防止录入自己声音
#
#         try:
#             audio_file_count += 1
#             # pyttsx3 保存也是 wav 格式比较稳
#             filename = os.path.join(folder_path, f"stream_{audio_file_count}.wav")
#
#             print(f"🔊 [pyttsx3] 正在合成: {text}")
#
#             # --- 核心合成逻辑 ---
#             # pyttsx3 的 save_to_file 是同步的，且非常快，直接运行即可
#             # 如果为了极致的线程安全，也可以放入 executor，但通常不需要
#             try:
#                 engine.save_to_file(text, filename)
#                 engine.runAndWait()  # 必须调用这个，文件才会生成
#             except Exception as e:
#                 print(f"pyttsx3 合成错误: {e}")
#                 continue
#
#             # 播放前打点
#             t3_play = time.time()
#             if t0_ref:
#                 latency = t3_play - t0_ref
#                 print(f"🚀 [T3] pyttsx3 响应延迟: {latency:.3f}s")
#
#             # 播放
#             if os.path.exists(filename):
#                 await async_play_audio(filename)
#
#                 # 删除临时文件
#                 try:
#                     os.remove(filename)
#                 except:
#                     pass
#
#         except Exception as e:
#             print(f"TTS/播放出错: {e}")
#         finally:
#             tts_text_queue.task_done()
#             if tts_text_queue.empty():
#                 is_speaking = False


# async def tts_consumer_worker():
#     """
#     后台任务：不断从 tts_text_queue 获取文字 -> CosyVoice 合成 -> 播放
#     """
#     global is_speaking, audio_file_count
#
#     while True:
#         item = await tts_text_queue.get()
#         if isinstance(item, tuple):
#             text, t0_ref = item
#         else:
#             text, t0_ref = item, None
#
#         # 过滤掉空的或者太短的文本，避免报错
#         if not text or len(text.strip()) < 1:
#             tts_text_queue.task_done()
#             continue
#
#         is_speaking = True
#         segments_to_save.clear()
#
#         try:
#             audio_file_count += 1
#             # CosyVoice 输出是 wav 格式，建议用 .wav 后缀
#             filename = os.path.join(folder_path, f"stream_{audio_file_count}.wav")
#
#             print(f"🔊 正在合成: {text}")
#
#             # --- CosyVoice 推理 (这是同步代码，且耗时，必须放入线程池) ---
#             loop = asyncio.get_event_loop()
#
#             # 定义一个同步函数来执行推理和保存
#             def run_cosyvoice_sync(input_text, out_path):
#                 # stream=False: 咱们是按句子进来的，直接生成整句比流式切片处理简单且效果好
#                 # '中文女' 是音色名，你可以改成 '中文男' 或其他
#                 model_output = cosyvoice_model.inference_sft(input_text, '中文女', stream=False)
#
#                 # 遍历生成器 (其实只有一段音频，因为 stream=False)
#                 for i, j in enumerate(model_output):
#                     # j['tts_speech'] 是 tensor, 采样率通常是 22050
#                     torchaudio.save(out_path, j['tts_speech'], 22050)
#                     return True  # 生成成功
#                 return False
#
#             # 在 executor 中运行，不阻塞主线程
#             success = await loop.run_in_executor(None, run_cosyvoice_sync, text, filename)
#
#             if success:
#                 # 播放前打点
#                 t3_play = time.time()
#                 if t0_ref:
#                     latency = t3_play - t0_ref
#                     print(f"🚀 [T3] CosyVoice 响应延迟: {latency:.3f}s")
#
#                 # 播放
#                 await async_play_audio(filename)
#
#                 # 删除
#                 try:
#                     os.remove(filename)
#                 except:
#                     pass
#             else:
#                 print("CosyVoice 合成未返回数据")
#
#         except Exception as e:
#             print(f"TTS/播放出错: {e}")
#         finally:
#             tts_text_queue.task_done()
#             if tts_text_queue.empty():
#                 is_speaking = False

# async def tts_consumer_worker():
#     """
#     后台任务：不断从 tts_text_queue 获取文字 -> 合成 -> 播放
#     实现了“边生成边播放”的效果，且保证句子顺序
#     """
#     global is_speaking, audio_file_count
#
#     while True:
#         # 等待队列中有文字
#         item = await tts_text_queue.get()
#         if isinstance(item, tuple):
#             text, t0_ref = item
#         else:
#             text, t0_ref = item, None
#
#         # text = await tts_text_queue.get()
#
#         # 标记正在说话
#         is_speaking = True
#         # 清空录音缓存，防止把自己说话录进去
#         segments_to_save.clear()
#
#         try:
#             audio_file_count += 1
#             filename = os.path.join(folder_path, f"stream_{audio_file_count}.mp3")
#
#             print(f"🔊 正在播报: {text}")
#
#             # 1. 生成语音 (Edge-TTS 是异步的)
#             communicate = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural")
#             await communicate.save(filename)
#
#             # 播放前打点
#             t3_play = time.time()  # <---【埋点 T3】开始播放
#             if t0_ref:
#                 latency = t3_play - t0_ref
#                 print(f"🚀 [T3] ！！！首字音频响应延迟！！！: {latency:.3f}s")
#
#             # 2. 播放语音 (必须await播放完成，否则下一句会重叠)
#             await async_play_audio(filename)
#
#             # 3. 删除临时文件
#             try:
#                 os.remove(filename)
#             except:
#                 pass
#
#         except Exception as e:
#             print(f"TTS/播放出错: {e}")
#         finally:
#             tts_text_queue.task_done()
#
#             # 如果队列空了，说明这一轮说话结束
#             if tts_text_queue.empty():
#                 is_speaking = False


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
            print(f"⏱️ [T0] 开始处理: {t0_start}")
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

        is_processing = True
        segments_to_save.clear()

        try:
            # --- 0. 首次运行注册逻辑 ---
            existing_users = glob.glob(os.path.join(set_SV_enroll, "*.wav"))
            if flag_sv_used and not existing_users and not flag_sv_enroll:
                print("👋 初次见面，请注册。")
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
            print(f"⏱️ [T1] ASR耗时: {t1_asr - t0_start:.3f}s")
            if not raw_text:
                is_processing = False
                continue

            pinyin_text = extract_pinyin(raw_text)
            print(f"👂 听到: {raw_text}")

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
                        t2_llm_first = time.time()  # <---【埋点 T2】LLM生成首句
                        print(f"⏱️ [T2] LLM首句生成耗时: {t2_llm_first - t1_asr:.3f}s")
                        print(f"📝 首句内容: {sentence}")
                        # 将 t0 传给 TTS 队列以便计算总延迟 (可选，或使用全局变量)
                        # 这里为了简单，我们只在 TTS 侧打印当前时间

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

    # 等待任务 (实际上是无限循环)
    await asyncio.gather(tts_task, inference_task)


if __name__ == "__main__":
    try:
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        recording_active = False
        print("系统退出")