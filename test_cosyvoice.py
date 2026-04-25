# python
# 诊断脚本：列出模型目录、尝试加载 spk2info.pt，并打印 cosyvoice 可用音色
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from funasr import AutoModel
import torchaudio
import pygame
import time
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

import yaml  # 显式导入yaml
import ruamel.yaml  # 导入ruamel.yaml


model_dir = r"D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\iic\CosyVoice-300M"
print("检查目录:", model_dir)
if os.path.exists(model_dir):
    for p in sorted(os.listdir(model_dir)):
        print("-", p)
else:
    print("模型目录不存在，请检查路径")

spk_path = os.path.join(model_dir, "spk2info.pt")
print("\n尝试加载:", spk_path)
if os.path.exists(spk_path):
    try:
        data = torch.load(spk_path, map_location="cpu")
        print("加载成功，类型:", type(data))
        if isinstance(data, dict):
            print("keys:", list(data.keys())[:50])
        else:
            # 打印部分内容以便诊断
            print("内容示例:", str(data)[:500])
    except Exception as e:
        print("加载 spk2info.pt 失败:", e)
else:
    print("未找到 spk2info.pt，检查文件名或是否放在子目录（如 `speakers/`）")


cosyvoice = CosyVoice(r'D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\iic\CosyVoice-300M', load_jit=True, load_onnx=False, fp16=True)
# --- CosyVoice - 支持的音色列表
print("支持的音色-----------------: ", cosyvoice.list_avaliable_spks())

# 如果 cosyvoice 实例已在当前作用域，打印可用音色
try:
    print("\ncosyvoice.list_avaliable_spks():", cosyvoice.list_avaliable_spks())
except Exception as e:
    print("调用 cosyvoice.list_avaliable_spks() 出错:", e)
