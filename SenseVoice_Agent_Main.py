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

# --- 导入我们的大脑 ---
from SenseVoice_Agent_Brain import SmartAgentBrain

# --- 配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
VAD_MODE = 3
OUTPUT_DIR = "./output"
NO_SPEECH_THRESHOLD = 1
folder_path = "./Test_Agent/"

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
set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'

# --- 初始化模型 ---
print("正在初始化模型，请稍候...")

# 1. 初始化 VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# 2. 初始化 SenseVoice (ASR)
# 请确保你的模型路径正确，或者使用 modelscope 自动下载的路径
model_dir = r"D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\ASR"
model_senceVoice = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")

# 3. 初始化 CAM++ (声纹)
sv_pipeline = pipeline(
    task='speaker-verification',
    model='D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\iic\CAM++',
    model_revision='v1.0.0',
    device="cuda:0"
)

# 4. 初始化 Agent 大脑 (连接 Milvus 和 LLM)
agent_brain = SmartAgentBrain()

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
    print(f"🤖 Agent: {text}")
    audio_file_count += 1
    filename = os.path.join(folder_path, f"reply_{audio_file_count}.mp3")
    asyncio.run(text_to_speech(text, filename))
    play_audio(filename)

    time.sleep(0.3)
    is_speaking = False


# --- 核心推理线程 ---
def Inference(audio_path):
    global flag_sv_enroll, flag_KWS, flag_KWS_used, flag_sv_used, set_SV_enroll, is_processing, segments_to_save

    is_processing = True  # 开始处理，暂停录音
    segments_to_save.clear()
    try:
        # 0. 检查声纹文件夹是否为空 (初次运行逻辑)
        if flag_sv_used and not os.path.exists(os.path.join(set_SV_enroll, "enroll_0.wav")):
            print("未检测到声纹，进入注册模式...")
            system_speak("请先说一句话注册声纹，需超过三秒哦。")
            flag_sv_enroll = 1
            return

        # 1. ASR 语音识别
        try:
            res = model_senceVoice.generate(input=audio_path, cache={}, language="auto", use_itn=False)
            raw_text = res[0]['text'].split(">")[-1].strip()
            pinyin_text = extract_pinyin(raw_text)
            print(f"👂 听到: {raw_text} (拼音: {pinyin_text})")
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
                system_speak("我在呢, 主人!")
                return
            else:
                # 如果没唤醒，直接忽略
                if not flag_KWS:
                    print("未唤醒...")
                    return

        # 3. 声纹验证 (SV)
        if flag_sv_used:
            try:
                enroll_path = os.path.join(set_SV_enroll, "enroll_0.wav")
                score = sv_pipeline([enroll_path, audio_path])
                print(f"🔐 声纹得分: {score['score']}")

                if score['score'] < thred_sv:
                    system_speak("声纹验证失败，我不能听你的指令。")
                    flag_KWS = 0  # 重置唤醒
                    return
            except Exception as e:
                print(f"SV Error: {e}")
                return

        # 4. 调用 Agent 大脑处理 (核心结合点)
        # 使用 asyncio.run 在同步线程中调用异步逻辑
        reply = asyncio.run(agent_brain.process_user_query(raw_text))

        # 5. 播报结果
        system_speak(reply)
        pass
    finally:
        is_processing = False  # 处理完成，恢复录音

    # 交互完成后，可以选择重置唤醒状态 (需再次唤醒)，或者保持唤醒
    # flag_KWS = 0


# --- 录音线程 ---
def audio_recorder():
    global recording_active, last_active_time, segments_to_save, last_vad_end_time, audio_file_count, flag_sv_enroll

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                    frames_per_buffer=CHUNK)
    audio_buffer = []

    print("🎤 麦克风监听中...")

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
                save_path = os.path.join(set_SV_enroll, "enroll_0.wav")

            # 写入文件
            wf = wave.open(save_path, 'wb')
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b''.join([seg[0] for seg in segments_to_save]))
            wf.close()

            segments_to_save.clear()  # 清空缓存

            if flag_sv_enroll:
                print("声纹注册文件已保存。")
                flag_sv_enroll = 0
                system_speak("声纹注册成功！现在可以叫我了。")
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