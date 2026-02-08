# filename: agent_main.py
import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
from queue import Queue
import webrtcvad
import os
import asyncio
import pygame
import edge_tts
from funasr import AutoModel
from modelscope.pipelines import pipeline
from pypinyin import pinyin, Style
import re
from SpeakerManager import SpeakerManager
# --- 导入我们的大脑 ---
from SenseVoice_Agent_Brain import SmartAgentBrain
import glob
# --- 配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
VAD_MODE = 3
OUTPUT_DIR = "../output"
NO_SPEECH_THRESHOLD = 1
folder_path = "../Test_Agent/"

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 全局变量
audio_queue = Queue()
recording_active = True
segments_to_save = []
last_active_time = time.time()
last_vad_end_time = 0
audio_file_count = 0

# --- KWS & 声纹配置 ---
set_KWS = "xiao ming tong xue"  # 唤醒词拼音
flag_KWS = 0
flag_KWS_used = 1  # 是否开启唤醒词
flag_sv_used = 1  # 是否开启声纹
flag_sv_enroll = 0  # 是否处于注册模式
thred_sv = 0.30  # 声纹阈值

is_speaking = False  # 是否正在播放语音
is_processing = False  # 是否正在处理推理（ASR+LLM+TTS整个流程）

# 声纹路径
# set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
set_SV_enroll = r'.\SpeakerVerification_DIR\users\\'
temp_register_name = "" # 用于暂存即将注册的用户名
# --- 初始化模型 ---
print("正在初始化模型，请稍候...")

# 初始化 VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)


# 初始化 Agent 大脑 (连接 Milvus 和 LLM)
agent_brain = SmartAgentBrain()
model_senceVoice = agent_brain.local_model.funasr_model
sv_pipeline = agent_brain.local_model.CAM_model

# 初始化多用户管理器
spk_manager = SpeakerManager(set_SV_enroll, agent_brain.local_model.CAM_model, threshold=0.35)
print(">>> 模型加载完成！系统启动！ <<<")


# --- 辅助函数 ---
def extract_pinyin(input_string):
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', input_string)
    chinese_text = ''.join(chinese_chars)
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    return ' '.join([item[0] for item in pinyin_result])


def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
    except Exception as e:
        print(f"播放失败: {e}")


async def text_to_speech(text, output_file):
    """使用 Edge TTS 生成语音"""
    voice = "zh-CN-XiaoyiNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def system_speak(text):
    """同步包装的 TTS 播放函数"""
    global audio_file_count, is_speaking, segments_to_save

    is_speaking = True
    segments_to_save.clear() # 清空之前的缓存，避免录入播报声音
    print(f"Agent Output: {text}")
    audio_file_count += 1
    filename = os.path.join(folder_path, f"reply_{audio_file_count}.mp3")
    asyncio.run(text_to_speech(text, filename))
    play_audio(filename)

    time.sleep(0.3)
    is_speaking = False


# --- 核心推理线程 ---
def Inference(audio_path):
    global flag_sv_enroll, flag_KWS, flag_KWS_used, flag_sv_used, set_SV_enroll, is_processing, segments_to_save, temp_register_name

    is_processing = True  # 开始处理，暂停录音
    segments_to_save.clear()
    current_user_id = "Guest"  # 默认为访客
    try:
        # 0. 检查声纹文件夹是否为空 (初次运行逻辑)
        existing_users = glob.glob(os.path.join(set_SV_enroll, "*.wav"))
        if flag_sv_used and not existing_users:
            print("声纹库为空，进入首个用户注册模式...")
            system_speak("欢迎使用，我需要先认识你。请说一句话大于3s的句子用于注册声纹。")
            temp_register_name = "主人"  # 默认第一个人叫主人
            flag_sv_enroll = 1
            return

        # 1. ASR 语音识别
        try:
            res = model_senceVoice.generate(input=audio_path, cache={}, language="auto", use_itn=False)
            raw_text = res[0]['text'].split(">")[-1].strip()
            pinyin_text = extract_pinyin(raw_text)
            print(f"听到: {raw_text} (拼音: {pinyin_text})")
        except Exception as e:
            print(f"ASR Error: {e}")
            return

        if not raw_text: return

        # 2. 唤醒词检测 (KWS)
        if flag_KWS_used:
            if set_KWS in pinyin_text:
                print(">>> 唤醒词匹配成功！")
                flag_KWS = 1
                # 唤醒成功, 播报
                system_speak("你好, 我在呢!")
                return
            else:
                # 如果没唤醒，直接忽略
                if not flag_KWS:
                    print("未唤醒...")
                    return
        # 2. 声纹对比 (CAM)
        if flag_sv_used:
            try:
                identified_user, score = spk_manager.identify(audio_path)

                if identified_user == "Unknown":
                    system_speak("身份验证失败，我不认识你。")
                    flag_KWS = 0
                    return  # 拒绝执行

                current_user_id = identified_user
                print(f"识别成功，当前用户: {current_user_id}")

            except Exception as e:
                print(f"SV Error: {e}")
                return

        # 4. 调用 Agent 处理
        # 使用 asyncio.run 在同步线程中调用异步逻辑
        reply = asyncio.run(agent_brain.process_user_query(raw_text, user_id=current_user_id))

        # 检查是否是注册指令
        if reply.startswith("ACTION_REGISTER:"):
            target_name = reply.split(":")[1]
            if target_name == "Unknown_User":
                system_speak("好的，请告诉我你怎么称呼？")
            else:
                temp_register_name = target_name
                flag_sv_enroll = 1  # 开启注册模式
                system_speak(f"好的，准备录入【{target_name}】的声纹。请在听到‘滴’声后，清晰地说一句话，至少3秒。")
                # 说完后, 进入下一轮 audio_recorder
            return

        # 5. 播报结果
        system_speak(reply)
        pass
    finally:
        is_processing = False  # 处理完成，恢复录音

    # 交互完成后，可以选择重置唤醒状态 (需再次唤醒)，或者保持唤醒
    # flag_KWS = 0


# --- 录音线程 ---
def audio_recorder():
    global recording_active, last_active_time, segments_to_save, last_vad_end_time, audio_file_count, flag_sv_enroll, temp_register_name

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                    frames_per_buffer=CHUNK)
    audio_buffer = []

    print("麦克风监听中...")

    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

        # 每 0.5 秒检测 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            # 简单 VAD 检测
            is_speech = False
            step = int(AUDIO_RATE * 0.02)
            speech_frames = 0
            for i in range(0, len(raw_audio), step):
                chunk = raw_audio[i:i + step]
                if len(chunk) == step and vad.is_speech(chunk, AUDIO_RATE):
                    speech_frames += 1
            if speech_frames > 5: is_speech = True

            if is_speech:
                # 正在处理或播报时不记录语音段
                if not is_speaking and not is_processing:
                    last_active_time = time.time()
                    segments_to_save.append((raw_audio, time.time()))

            audio_buffer = []

        # 判定句子结束 (静音超时)
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD and segments_to_save:

            # 正在播报，跳过处理并清空缓存
            if is_speaking or is_processing:
                segments_to_save.clear()
                continue
            # 保存并推理
            audio_file_count += 1

            # 处理声纹注册的特殊逻辑
            save_path = f"{OUTPUT_DIR}/audio_tmp.wav"
            if flag_sv_enroll:
                os.makedirs(set_SV_enroll, exist_ok=True)
                # 如果没有名字，就用时间戳兜底
                if not temp_register_name:
                    final_name = f"User_{int(time.time())}.wav"
                else:
                    final_name = f"{temp_register_name}.wav"

                save_path = os.path.join(set_SV_enroll, final_name)

            # 写入文件
            wf = wave.open(save_path, 'wb')
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b''.join([seg[0] for seg in segments_to_save]))
            wf.close()

            segments_to_save.clear()  # 清空缓存

            if flag_sv_enroll:
                print(f"声纹注册文件已保存: {save_path}")
                flag_sv_enroll = 0  # 关闭开关
                temp_register_name = ""  # 清空暂存名
                # SpeakerManager 刷新用户列表
                try:
                    spk_manager.refresh_speakers()
                except:
                    pass

                system_speak("注册成功！我已经记住你的声音了。")
            else:
                # 开启新线程推理，避免阻塞录音
                t = threading.Thread(target=Inference, args=(save_path,))
                t.start()

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    try:
        t_rec = threading.Thread(target=audio_recorder)
        t_rec.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recording_active = False