import httpx


def test_audio_chat():
    url = "http://127.0.0.1:8000/api/chat-audio"
    file_path = "test_audio.wav"  # 确保你本地有一个测试音频

    with open(file_path, "rb") as f:
        files = {"audio": (file_path, f, "audio/mpeg")}
        data = {"user_id": "TestUser_001"}

        # 使用 httpx 的流式请求
        with httpx.stream("POST", url, files=files, data=data, timeout=None) as r:
            print(f"状态码: {r.status_code}")
            for line in r.iter_lines():
                if line:
                    print(f"收到数据: {line}")


if __name__ == "__main__":
    test_audio_chat()