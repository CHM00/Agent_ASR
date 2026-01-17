from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from funasr import AutoModel
from modelscope.pipelines import pipeline


class Load_Model:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model_path = r"D:\Qwen"
        self.funasr_model_path = r"D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\ASR"
        self.CAM_model_path = r"D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\iic\CAM++"

        self.llm_model, self.tokenizer = self.load_local_llm()
        self.funasr_model = self.load_funasr()
        self.CAM_model = self.load_cam()

    # 1. 全局加载本地模型和Tokenizer（只加载一次，避免重复耗时）
    def load_local_llm(self):
        # 加载模型：自动适配设备（CPU/GPU）、自动 dtype
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype="auto",
            device_map="auto",  # 自动分配到GPU/CPU
            trust_remote_code=True  # Qwen 需要开启
        )
        # 加载分词器：必须和模型匹配（同一仓库/版本）
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True,
            padding_side="right"  # 避免生成时的警告
        )
        # 补充特殊标记（部分开源模型需要）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


    def load_funasr(self):
        model_dir = self.funasr_model_path
        model_senceVoice = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")
        return model_senceVoice

    def load_cam(self):
        # 3. 初始化 CAM++ (声纹)
        sv_pipeline = pipeline(
            task='speaker-verification',
            model=self.CAM_model_path,
            model_revision='v1.0.0',
            device="cuda:0"
        )
        return sv_pipeline


    # 2. 本地推理
    def llm_chat(self, messages, system_prompt=None):
        # 构建标准化对话模板（适配 Qwen 的格式）
        # Step 1: Tokenizer 编码（文本 → 数字ID）
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 追加助手回复的起始标记
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # 适配模型最大长度
        ).to(self.device)  # 输入张量移到模型所在设备（GPU/CPU）

        # Step 2: 本地模型推理（数字ID → 生成的数字ID）
        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=512,  # 限制生成长度
            temperature=0.7,  # 随机性（越低越固定）
            top_p=0.9,  # 采样策略
            do_sample=True,  # 开启采样（否则是贪心生成）
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Step 3: Tokenizer 解码（数字ID → 文本）
        # 只截取生成的部分（排除输入的ID），过滤特殊标记
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("本地模型输出: ", output_text)
        return output_text