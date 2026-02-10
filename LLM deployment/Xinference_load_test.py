import json
import time
from locust import HttpUser, task, between, events

MODEL_UID = "qwen3"
BASE_URL = "http://localhost:6006" # autodl的公网端口映射


def get_chat_payload():
    return {
        "model": MODEL_UID,
        "messages": [
            {"role": "user", "content": "请简要解释一下人工智能的核心原理，要求 50 字以内"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False  # 非流式输出，便于统计响应时间
    }

class XinferenceUser(HttpUser):
    # 模拟用户思考时间：1-3 秒
    wait_time = between(1, 3)

    base_url = BASE_URL

    # 核心压测任务：调用模型的 chat/completions 接口
    @task(1)
    def chat_completion(self):
        start_time = time.time()
        try:
            # 发送 POST 请求调用模型
            response = self.client.post(
                "/v1/chat/completions",
                json=get_chat_payload(),
                headers={"Content-Type": "application/json"},
                timeout=60  # 超时时间 60 秒（避免高并发下请求卡死）
            )
            # 记录成功请求的响应时间
            response_time = (time.time() - start_time) * 1000  # 转毫秒
            events.request.fire(
                request_type="POST",
                name="chat_completions",
                response_time=response_time,
                response_length=len(response.content),
                exception=None
            )
        except Exception as e:
            # 记录失败请求
            events.request.fire(
                request_type="POST",
                name="chat_completions",
                response_time=(time.time() - start_time) * 1000,
                response_length=0,
                exception=e
            )

# 压测启动入口（命令行启动）
if __name__ == "__main__":
    '''
    启动命令示例：
        (1) 50 并发用户，每秒启动 5 个，压测 3 分钟，结果会保存到 xinference_4b_test_*.csv 文件中
        locust -f xinference_load_test.py --host=http://localhost:6006 \
                    --users 50 --spawn-rate 5 --run-time 3m --headless \
                    --csv=xinference_4b_test
                    
        (2) 500 并发用户，每秒启动 10 个，压测 3 分钟，结果会保存到 xinference_4b_test_*.csv 文件中
        locust -f xinference_load_test.py --host=http://localhost:6006 \
                    --users 500 --spawn-rate 10 --run-time 3m --headless \
                    --csv=xinference_4b_stress_test
        参数说明: 
            --users 设置最大并发用户数为 500   
            --spawn-rate 10 设置每秒生成用户的速度, 系统不会瞬间启动 500 个用户，而是每秒增加 10 个用户，直到达到 500 为止。这样可以观察系统性能随压力增加而演变的曲线。
            --run-time 10m 压测持续10分钟
            --csv=xinference_stress_test 将测试结果持久化保存为 CSV 文件
    '''
    import os
    # 启动时指定 Xinference 的 6006 端口
    os.system("locust -f xinference_load_test.py --host=http://localhost:6006 --web-host=0.0.0.0")