

# 大模型本地部署与性能压测实战 (AutoDL + Xinference/vLLM)

本项目记录了在 AutoDL 平台上，利用 **4090D (24G)** 显卡进行大模型（Qwen3 系列）本地部署的全过程，涵盖了 **Xinference** 和 **vLLM** 两种主流后端，并附带了基于 **Locust** 的并发压力测试分析。

---

## 🛠️ 环境配置

### 1. 基础环境准备
在 AutoDL 的数据盘（`/root/autodl-tmp`）中创建隔离的 Conda 环境：

```bash
cd /root/autodl-tmp

# 创建并激活环境
conda create -n llm_deploy python=3.12 -y
conda activate llm_deploy

# 安装推理框架
pip install "xinference[vllm,transformers]" locust openai

```

### 2. 加速与路径优化

配置环境变量以使用 **ModelScope (魔搭)** 镜像加速模型下载，并指定数据盘存储路径：

```bash
# 设置模型存储路径
export XINFERENCE_HOME=/root/autodl-tmp/xinference
# 切换模型源为 ModelScope (国内访问加速)
export XINFERENCE_MODEL_SRC=modelscope

# 写入配置文件永久生效
echo 'export XINFERENCE_HOME=/root/autodl-tmp/xinference' >> ~/.bashrc
echo 'export XINFERENCE_MODEL_SRC=modelscope' >> ~/.bashrc
source ~/.bashrc

```

---

## 🚀 部署方案一：Xinference

Xinference 提供了便捷的 GUI 界面和多引擎支持。

### 1. 启动服务

```bash
# 建议后台运行并记录日志
nohup xinference-local --host 0.0.0.0 --port 6006 > xinference.log 2>&1 &

```

### 2. 模型配置要点

在 Web 界面部署 `Qwen3-4B-fp8` 时，关键参数建议如下：

* **Engine**: 选择 `vLLM` 以获得最佳吞吐。
* **gpu_memory_utilization**: `0.70` (预留部分显存用于 KV Cache 动态增长)。
* **max_model_len**: `2048` 或 `4096` (减小此值可显著降低显存占用，提升并发能力)。

---

## 📈 性能压测分析 (Locust)

针对 `Qwen3-4B-fp8` 模型，在输出 Token 固定为 50 的场景下进行压力测试：

### 并发性能对比表

| 指标 | 50 并发 | 500 高并发 | 变化结论 |
| --- | --- | --- | --- |
| **每秒吞吐 (RPS)** | 30.71 | 91.77 | 吞吐量随并发大幅增长，GPU 利用率饱和 |
| **平均响应时间** | 1.18 秒 | 3.21 秒 | 延迟增加，系统开始排队 |
| **95% 分位延迟** | 1.20 秒 | 3.70 秒 | 高并发下用户体验下降约 2 秒 |

> **长文本压力结论**：在输出 512 Token 且并发为 500 时，首字延迟 (TTFT) 飙升至约 15s。虽然总吞吐能达到约 **7500 Token/s**，但极长的排队时间意味着单卡 4090D 难以支撑 500 个用户的长文本实时对话。

---

## ⚡ 部署方案二：vLLM 原生部署

vLLM 是目前最高效的推理引擎之一，支持 BF16 精度和 DeepSeek 式的推理解析。

### 1. 启动命令

```bash
vllm serve /root/autodl-tmp/model/Qwen/Qwen3-4B-Instruct-2507 \
    --served-model-name Qwen3-4B \
    --max_model_len 1024 \
    --reasoning-parser deepseek_r1 \
    --host 0.0.0.0 \
    --port 6006

```

### 2. Python 客户端调用 (OpenAI SDK)

```python
from openai import OpenAI

# 使用 AutoDL 提供的公网映射地址
client = OpenAI(
    api_key="EMPTY", 
    base_url="https://<your-autodl-url>:8443/v1"
)

response = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "你好，请开始推理"}]
)

print(f"Content: {response.choices[0].message.content}")
# 如果使用了推理模型，可打印思考过程
print(f"Reasoning: {response.choices[0].message.reasoning_content}")

```

---

## 💡 总结与建议

1. **显存管理**：4090D (24G) 在部署 4B 模型时非常从容，但并发增加时 KV Cache 是主要瓶颈，务必通过 `max_model_len` 进行平衡。
2. **引擎选择**：追求稳定性与易用性选 **Xinference**；追求极限性能与原生 API 兼容性选 **vLLM**。
3. **网络优化**：在 AutoDL 部署时，利用 `modelscope` 环境变量可以节省数小时的模型下载时间。

参考个人博客链接:
https://blog.csdn.net/weixin_49891405/article/details/157938510?spm=1001.2014.3001.5501

##  Acknowledgments
DataWhale AI开源组织 (https://github.com/datawhalechina)

