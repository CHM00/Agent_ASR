from openai import OpenAI

# 这里的地址换成 AutoDL 提供的公网 URL 和 6006 端口
client = OpenAI(api_key="EMPTY", base_url="https://u514074-be32-f5ef0b3a.westb.seetacloud.com:8443/v1")

response = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "你好，请开始推理"}]
)
print(response)
print(response.choices[0].message.reasoning_content)