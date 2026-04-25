import os
import glob

class SpeakerManager:
    def __init__(self, enroll_dir, model_pipeline, threshold=0.30):
        self.enroll_dir = enroll_dir
        self.model = model_pipeline
        self.threshold = threshold
        # 预加载所有注册用户
        # 结构: {"user_name": "path/to/wav"}
        self.speakers = {}
        self.refresh_speakers()

    def refresh_speakers(self):
        """扫描目录加载用户"""
        os.makedirs(self.enroll_dir, exist_ok=True)
        wav_files = glob.glob(os.path.join(self.enroll_dir, "*.wav"))
        self.speakers = {}
        for f in wav_files:
            # 文件名作为用户名，例如 "Dad.wav" -> 用户名 "Dad"
            name = os.path.splitext(os.path.basename(f))[0]
            self.speakers[name] = f
        print(f"👥 已加载声纹库: {list(self.speakers.keys())}")

    def identify(self, audio_path):
        """1:N 识别用户"""
        print("self.speakers: ", self.speakers)
        if not self.speakers:
            return "Unknown", 0.0

        best_score = -1
        best_user = "Unknown"

        # 遍历对比 (对于家庭场景 <20 人，遍历效率足够)
        for user, enroll_path in self.speakers.items():
            try:
                # CAM++ 接受 [enroll, test] 列表
                res = self.model([enroll_path, audio_path])
                score = res['score']
                if score > best_score:
                    best_score = score
                    best_user = user
            except Exception as e:
                print(f"对比 {user} 失败: {e}")
                continue

        print(f"🔍 声纹识别结果: Top1={best_user}, Score={best_score:.4f}")

        if best_score > self.threshold:
            return best_user, best_score
        else:
            return "Unknown", best_score