import os
from openai import OpenAI

# 这里的地址换成 AutoDL 提供的公网 URL 和 6006 端口
base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "你好，请开始推理"}]
)
print(response)
print(response.choices[0].message.reasoning_content)