import re
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertIntentRouter:
    def __init__(self, model_dir: str, device: str = None):
        self.labels = ["call_elm", "need_search", "register", "chat"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 可按业务扩充
        self.search_kw = ["天气", "新闻", "百科", "价格", "汇率", "股价", "几点", "怎么", "是什么"]
        self.food_kw = ["想吃", "点", "来一份", "外卖", "饿了", "菜单"]
        self.register_kw = ["注册", "录入", "录声音", "声纹", "我是"]

    @torch.no_grad()
    def predict_intent(self, text: str):
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
        print("pred_id:", pred_id)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[Latency] BERT predict_intent: {dt_ms:.2f} ms")
        return self.labels[pred_id]

    def extract_slots(self, text: str, intent: str):
        result = {
            "Call_elm": False,
            "Food_candidate": "",
            "Need_Search": "",
            "Register_Action": ""
        }

        if intent == "call_elm":
            result["Call_elm"] = True
            # 简单规则抽取食物名（可换成NER）
            m = re.search(r"(想吃|来一份|点)(.+?)(吗|吧|。|！|？|$)", text)
            if m:
                result["Food_candidate"] = m.group(2).strip()
            else:
                result["Food_candidate"] = text.strip()

        elif intent == "need_search":
            # 搜索词直接用原句或去停用词后拼接
            result["Need_Search"] = text.strip()

        elif intent == "register":
            # 抽取“我是张三”
            m = re.search(r"我是([\u4e00-\u9fa5A-Za-z0-9_]{1,16})", text)
            result["Register_Action"] = m.group(1) if m else "Unknown_User"
        print("BERT提取的result: ", result)
        return result

    def route(self, text: str):
        t0 = time.perf_counter()
        bert_ms=0.0
        # 关键词先验，进一步降延迟并增强鲁棒性
        if any(k in text for k in self.register_kw):
            intent = "register"
        elif any(k in text for k in self.food_kw):
            intent = "call_elm"
        elif any(k in text for k in self.search_kw):
            intent = "need_search"
        else:
            intent = self.predict_intent(text)

        result = self.extract_slots(text, intent)
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"[Latency] Intent route total: {total_ms:.2f} ms")
        return result
