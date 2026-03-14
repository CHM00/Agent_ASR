import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report
from datasets import ClassLabel


LABEL2ID = {"call_elm": 0, "need_search": 1, "register": 2, "chat": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 加载数据
df = pd.read_json("./train_data.json")

dataset = Dataset.from_dict(df.to_dict("list"))


dataset = dataset.cast_column("label", ClassLabel(num_classes=4, names=["call_elm", "need_search", "register", "chat"]))

# 划分训练集 / 验证集
split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"训练集: {len(train_dataset)} 条 | 验证集: {len(eval_dataset)} 条")

# 本地加载模型和分词器
model_name = "./rbt3"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=32)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset  = eval_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 训练参数
training_args = TrainingArguments(
    output_dir="./intent_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,           # 数据量小，适当增加轮数
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,   # 训练结束后自动加载最优 checkpoint
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# 验证集
predictions = trainer.predict(eval_dataset)
preds  = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\n分类结果")
print(classification_report(labels, preds, target_names=list(LABEL2ID.keys())))

# 保存模型
trainer.save_model("./final_intent_model")
tokenizer.save_pretrained("./final_intent_model")
print("训练完成！模型已保存至 ./final_intent_model")

# 推理
print("\n推理测试")
test_sentences = [
    "帮我点一份披萨",
    "今天上海天气怎么样",
    "我是新用户，请录入我的信息",
    "你好，陪我聊聊天吧",
]

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for text in test_sentences:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id    = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()
    print(f"  输入: {text}")
    print(f"  意图: {ID2LABEL[pred_id]}  置信度: {confidence:.2%}\n")