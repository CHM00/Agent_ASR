from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# 将JSON文件转换为CSV文件
file_path="formatted_finetune_data.jsonl"
df = pd.read_json(file_path, lines=True, encoding='utf-8') # 注意修改
ds = Dataset.from_pandas(df)


# 加载模型 tokenizer
tokenizer = AutoTokenizer.from_pretrained('D:\Qwen', trust_remote=True)

# 打印一下 chat template
messages = [
    {"role": "system", "content": "===system_message_test==="},
    {"role": "user", "content": "===user_message_test==="},
    {"role": "assistant", "content": "===assistant_message_test==="},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
print(text)


def process_func(example):
    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], []

    # 1. 提取对话内容
    # 假设每条数据都是从 system 开始，然后 human/gpt 交替
    convs = example['conversations']

    # 2. 构造符合 Qwen3 ChatML 格式的输入
    # Qwen3 依然遵循 <|im_start|> 这种 Prompt 格式
    instruction_text = ""
    for msg in convs:
        role = msg['from']
        content = msg['value']
        if role == 'system':
            instruction_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == 'human':
            # 如果是最后一轮对话前的 human，加入输入
            instruction_text += f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        elif role == 'gpt':
            # 注意：在微调中，我们通常需要模型预测 gpt 的内容
            # 这里构造一个“预答复”格式
            response_text = f"{content}"

            # 对指令部分进行编码
            instruction_encoded = tokenizer(instruction_text, add_special_tokens=False)
            # 对回复部分进行编码
            response_encoded = tokenizer(response_text, add_special_tokens=False)

            # 拼接
            input_ids = instruction_encoded["input_ids"] + response_encoded["input_ids"] + [tokenizer.pad_token_id]
            attention_mask = instruction_encoded["attention_mask"] + response_encoded["attention_mask"] + [1]

            # 标签设置：指令部分用 -100 掩码，只计算 response 的 loss
            labels = [-100] * len(instruction_encoded["input_ids"]) + response_encoded["input_ids"] + [tokenizer.pad_token_id]

            # # Qwen3 通常在每条数据末尾需要 pad 或者 eos
            # input_ids.append(tokenizer.eos_token_id)
            # attention_mask.append(1)
            # labels.append(tokenizer.eos_token_id)

            break  # 针对 SFT，通常每条样本只取一个对话对进行训练，或者你可以根据需求调整

    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def preprocess_multi_turn_qwen(example):

    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], []

    # 1. 提取对话内容
    # 假设每条数据都是从 system 开始，然后 human/gpt 交替
    convs = example['conversations']

    # 2. 构造符合 Qwen3 ChatML 格式的输入
    # Qwen3 依然遵循 <|im_start|> 这种 Prompt 格式
    # instruction_text = ""

    for msg in convs:
        role = msg["from"]
        content = msg["value"]

        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        # 1. 构造角色前缀: <|im_start|>system\n 或 <|im_start|>user\n 等
        prefix = f"<|im_start|>{role}\n"
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

        # 2. 构造内容: {content}
        content_ids = tokenizer(content, add_special_tokens=False)["input_ids"]

        # 3. 构造后缀: <|im_end|>\n
        suffix = "<|im_end|>\n"
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

        # 拼接当前轮次的完整 ID
        current_turn_ids = prefix_ids + content_ids + suffix_ids
        input_ids.extend(current_turn_ids)

        # 设置 Labels: 只有角色为 assistant 时，content 和 suffix 部分才计算 Loss
        if role == "assistant":
            # 前缀部分不学 (-100)
            turn_labels = [-100] * len(prefix_ids)
            # 内容和结束符要学 (content_ids + suffix_ids)
            turn_labels += content_ids + suffix_ids
            labels.extend(turn_labels)
        else:
            # system 和 user 的所有部分都不学 (-100)
            labels.extend([-100] * len(current_turn_ids))

    # 统一截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }



tokenized_ds = ds.map(preprocess_multi_turn_qwen, remove_columns=ds.column_names)
print("tokenized data:", tokenized_ds)

# 首先配置 LoRA 参数
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型为 CLM，即 SFT 任务的类型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 目标模块，即需要进行 LoRA 微调的模块
    inference_mode=False, # 训练模式
    r=2, # Lora 秩，即 LoRA 微调的维度
    lora_alpha=8, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


import torch
# 重新加载一个 Base 模型
model = AutoModelForCausalLM.from_pretrained('D:\Qwen', device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
# 通过下列代码即可向模型中添加 LoRA 模块
model = get_peft_model(model, config)
# 查看 lora 微调的模型参数
print("model parameter: ", model.print_trainable_parameters())

from swanlab.integration.transformers import SwanLabCallback

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen3_4B_lora", # 输出目录
    per_device_train_batch_size=8, # 每个设备上的训练批量大小
    gradient_accumulation_steps=4, # 梯度累积步数
    logging_steps=10, # 每10步打印一次日志
    num_train_epochs=2, # 训练轮数
    save_steps=100, # 每100步保存一次模型
    learning_rate=1e-4, # 学习率
    save_on_each_node=True, # 是否在每个节点上保存模型
    gradient_checkpointing=True, # 是否使用梯度检查点
    report_to="none", # 不使用任何报告工具
)
swanlab_callback = SwanLabCallback(
    project="Qwen3-4B-lora",
    experiment_name="Qwen3-4B-experiment"
)

# 然后同样使用 trainer 训练即可
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)
# 开始训练
trainer.train()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = '/root/autodl-tmp/model/Qwen/Qwen3-4B-Instruct-2507'# 基座模型参数路径
lora_path = './output/Qwen3_4B_lora/checkpoint-64' # 这里改成你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, rust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)