from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from funasr import AutoModel
from modelscope.pipelines import pipeline
from cosyvoice.cli.cosyvoice import CosyVoice
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os
import yaml
import ruamel.yaml
import torchaudio
# 修复 yaml 问题
if not hasattr(yaml.Loader, 'max_depth'):
    yaml.Loader.max_depth = None
if not hasattr(ruamel.yaml.Loader, 'max_depth'):
    ruamel.yaml.Loader.max_depth = None


@dataclass
class CompressionConfig:
    max_context_tokens: int = 1600
    max_recent_turns: int = 8
    retrieval_budget_tokens: int = 500
    history_summary_trigger_turns: int = 10
    summary_max_chars: int = 1000


class Load_Model:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model_path = r"D:\Qwen"
        self.funasr_model_path = r"D:\git_repo_file\AgentASR_HTTP\ASR"
        self.CAM_model_path = r"D:\git_repo_file\AgentASR_HTTP\iic\CAM++"
        self.cosyvoice_model_path = r"D:\git_repo_file\AgentASR_HTTP\iic\CosyVoice-300M"


        self.llm_model, self.tokenizer = self.load_local_llm()
        self.funasr_model = self.load_funasr()
        self.CAM_model = self.load_cam()
        self.cosyvoice_model = self.load_cosyvoice()



    # 全局加载本地模型和Tokenizer（只加载一次，避免重复耗时）
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


    def load_cosyvoice(self):
        try:
            # load_jit=True 加速推理，fp16=True 节省显存
            model = CosyVoice(self.cosyvoice_model_path, load_jit=True, load_onnx=False, fp16=True)
            print(f"CosyVoice 加载成功，支持音色: {model.list_avaliable_spks()}")
            return model
        except Exception as e:
            print(f"CosyVoice 加载失败: {e}")
            return None

    # 本地推理
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



    def estimate_tokens_from_text(self, text: str) -> int:
        """
        粗估 token；中文场景用字符长度近似，避免频繁 tokenizer 开销。
        """
        if not text:
            return 0
        return max(1, int(len(text) / 1.8))

    def estimate_tokens_from_messages(self, messages: List[Dict[str, str]]) -> int:
        if not messages:
            return 0
        joined = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
        return self.estimate_tokens_from_text(joined)

    def compress_text(self, text: str, max_chars: int = 700) -> str:
        """
        用当前本地 LLM 做摘要压缩，保留实体、时间、数字、结论、用户意图。
        """
        if not text:
            return ""
        if len(text) <= max_chars:
            return text

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "你是上下文压缩器。请在不丢失关键信息的前提下压缩文本。"
                    "必须保留：人物/地点/时间/数值/用户明确要求/待办事项。"
                    "输出中文，150-220字，禁止编造。"
                ),
            },
            {"role": "user", "content": text},
        ]
        summary = self.llm_chat(prompt_messages)
        if not summary:
            return text[:max_chars]
        return summary.strip()[:max_chars]

    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        summary_max_chars: int = 700
    ) -> Dict[str, Any]:
        """
        将多轮消息压缩成结构化摘要。
        """
        if not messages:
            return {"summary": "", "facts": [], "open_questions": []}

        raw = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "你是对话记忆压缩器。请输出三段："
                    "1) SUMMARY: 关键脉络；"
                    "2) FACTS: 稳定事实(列表)；"
                    "3) OPEN_QUESTIONS: 未完成问题(列表)。"
                    "禁止虚构。中文输出。"
                ),
            },
            {"role": "user", "content": raw},
        ]
        out = self.llm_chat(prompt_messages).strip()

        # 最小解析（容错，不依赖严格json）
        summary, facts, open_q = out, [], []
        lines = [x.strip("- ").strip() for x in out.splitlines() if x.strip()]
        current = None
        for ln in lines:
            u = ln.upper()
            if u.startswith("SUMMARY"):
                current = "summary"
                continue
            if u.startswith("FACTS"):
                current = "facts"
                continue
            if u.startswith("OPEN_QUESTIONS"):
                current = "open"
                continue
            if current == "facts":
                facts.append(ln)
            elif current == "open":
                open_q.append(ln)

        summary = summary[:summary_max_chars]
        return {"summary": summary, "facts": facts[:12], "open_questions": open_q[:12]}

    def build_context_window(
        self,
        system_prompt: str,
        recent_messages: List[Dict[str, str]],
        running_summary: str,
        retrieved_context: str,
        user_input: str,
        cfg: Optional[CompressionConfig] = None
    ) -> List[Dict[str, str]]:
        """
        按 token 预算组装上下文：
        system -> summary -> recent -> retrieval -> current user
        超预算时优先裁剪 retrieval，再压缩 recent。
        """
        cfg = cfg or CompressionConfig()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if running_summary:
            messages.append({"role": "system", "content": f"历史摘要:\n{running_summary}"})
        if retrieved_context:
            messages.append({"role": "system", "content": f"检索上下文:\n{retrieved_context}"})
        messages.extend(recent_messages[-cfg.max_recent_turns:])
        messages.append({"role": "user", "content": user_input})

        total = self.estimate_tokens_from_messages(messages)
        if total <= cfg.max_context_tokens:
            return messages

        # 先缩检索
        if retrieved_context:
            compressed_retrieval = self.compress_text(retrieved_context, max_chars=450)
            messages = [m for m in messages if not m["content"].startswith("检索上下文:")]
            messages.insert(1, {"role": "system", "content": f"检索上下文(压缩):\n{compressed_retrieval}"})

        total = self.estimate_tokens_from_messages(messages)
        if total <= cfg.max_context_tokens:
            return messages

        # 再缩历史 recent（保留最近2轮原文）
        keep_recent = recent_messages[-2:]
        old_recent = recent_messages[:-2]
        old_summary = ""
        if old_recent:
            old_summary = self.compress_messages(old_recent, summary_max_chars=350).get("summary", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        merged_summary = (running_summary + "\n" + old_summary).strip()
        if merged_summary:
            messages.append({"role": "system", "content": f"历史摘要:\n{merged_summary[:700]}"})
        if retrieved_context:
            messages.append({"role": "system", "content": f"检索上下文(压缩):\n{self.compress_text(retrieved_context, max_chars=350)}"})
        messages.extend(keep_recent)
        messages.append({"role": "user", "content": user_input})

        return messages
