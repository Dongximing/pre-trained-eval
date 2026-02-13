# DExperts 使用示例

## 快速开始

### 1. 基本配置 (`config.yaml`)

```yaml
models:
  base:
    path: "/storage/data/original_models/Qwen2.5-32B"
    devices: [0, 1]
  expert:
    path: "/storage/data/original_models/Qwen3-8B"
    devices: [2, 3]
    tokenizer_path: "/storage/data/original_models/Qwen3-8B"
  antiexpert:
    path: "/storage/data/original_models/Qwen3-8B-base"
    devices: [4, 5]

api:
  host: "0.0.0.0"
  port: 8402

generation:
  max_new_tokens: 16000
  temperature: 0.7
  top_k: 20
  top_p: 0.8
  batch_size: 1
  alpha: 1.0
```

### 2. 启动API服务器

```bash
cd /home/ximing/pre-trained-eval/proxy-tuning
python runapi.py
```

输出示例：
```
2026-02-12 18:30:00 - __main__ - INFO - Setting CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
2026-02-12 18:30:00 - __main__ - INFO - 🔄 Loading DExperts models...
2026-02-12 18:30:00 - __main__ - INFO -   Base model: /storage/data/original_models/Qwen2.5-32B on GPUs [0, 1]
2026-02-12 18:30:00 - __main__ - INFO -   Expert model: /storage/data/original_models/Qwen3-8B on GPUs [2, 3]
2026-02-12 18:30:00 - __main__ - INFO -   Anti-expert model: /storage/data/original_models/Qwen3-8B-base on GPUs [4, 5]
2026-02-12 18:30:45 - __main__ - INFO - ✅ DExperts models loaded successfully!
2026-02-12 18:30:45 - __main__ - INFO - 🚀 Starting server on 0.0.0.0:8402
```

## 使用场景

### 场景1: OpenAI兼容API调用

```bash
curl -X POST http://localhost:8402/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

**Python客户端示例**:
```python
import requests

url = "http://localhost:8402/v1/chat/completions"
data = {
    "messages": [
        {"role": "user", "content": "Write a sorting algorithm in Python"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9
}

response = requests.post(url, json=data)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

### 场景2: 使用OpenAI SDK (兼容)

```python
from openai import OpenAI

# 指向你的本地服务器
client = OpenAI(
    base_url="http://localhost:8402/v1",
    api_key="dummy"  # API key不需要，但SDK要求提供
)

response = client.chat.completions.create(
    model="dexperts",
    messages=[
        {"role": "user", "content": "Explain how binary search works"}
    ],
    max_tokens=500,
    temperature=0.8
)

print(response.choices[0].message.content)
```

### 场景3: 不同GPU配置示例

#### 场景3.1: 所有模型共享GPU（内存受限）
```yaml
models:
  base:
    path: "/path/to/base"
    devices: [0, 1]      # 使用GPU 0,1
  expert:
    path: "/path/to/expert"
    devices: [0, 1]      # 共享GPU 0,1
  antiexpert:
    path: "/path/to/anti"
    devices: [0, 1]      # 共享GPU 0,1
```

#### 场景3.2: 大模型+小模型配置
```yaml
models:
  base:
    path: "/path/to/Qwen2.5-72B"  # 大模型
    devices: [0, 1, 2, 3]          # 使用4个GPU
  expert:
    path: "/path/to/Qwen3-8B"      # 小模型
    devices: [4]                   # 单GPU
  antiexpert:
    path: "/path/to/Qwen3-8B-base"
    devices: [5]                   # 单GPU
```

#### 场景3.3: 多服务器配置（不同模型在不同机器）
```yaml
# 服务器1: 只运行base模型
models:
  base:
    path: "/path/to/base"
    devices: [0, 1, 2, 3]

# 服务器2: 运行expert和antiexpert
models:
  expert:
    path: "/path/to/expert"
    devices: [0, 1]
  antiexpert:
    path: "/path/to/anti"
    devices: [2, 3]
```

### 场景4: 调整生成参数

#### 更保守的生成（低温度）
```yaml
generation:
  temperature: 0.3  # 更确定性
  top_k: 10
  top_p: 0.9
```

#### 更有创造性的生成（高温度）
```yaml
generation:
  temperature: 1.0  # 更随机
  top_k: 50
  top_p: 0.95
```

#### 使用熵门控（智能切换）
```yaml
generation:
  use_entropy_gating: true
  entropy_threshold: 5.0  # base模型熵低于5时使用base
```

### 场景5: 直接使用Python代码（不启动服务器）

```python
from generation import load_dexperts_model_and_tokenizer, generate_completions

# 加载模型
model, tokenizer = load_dexperts_model_and_tokenizer(
    base_model_name_or_path="/path/to/Qwen2.5-32B",
    expert_model_name_or_path="/path/to/Qwen3-8B",
    antiexpert_model_name_or_path="/path/to/Qwen3-8B-base",
    base_devices=[0, 1],
    expert_devices=[2, 3],
    antiexpert_devices=[4, 5],
    alpha=1.0
)

# 生成
prompts = ["Write a hello world program in Python"]
outputs = generate_completions(
    model=model,
    tokenizer=tokenizer,
    prompts_an=(prompts, [""]),
    batch_size=1,
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9
)

print(outputs[0])
```

### 场景6: 批量生成（提高效率）

```python
# config.yaml 中设置
generation:
  batch_size: 4  # 每次处理4个prompt

# Python代码
prompts = [
    "Write a sorting algorithm",
    "Explain binary search",
    "Implement a linked list",
    "Create a hash table"
]

outputs = generate_completions(
    model=model,
    tokenizer=tokenizer,
    prompts_an=(prompts, [""] * len(prompts)),
    batch_size=4,  # 批处理
    max_new_tokens=500
)

for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {output}\n")
```

### 场景7: 多端口部署（避免冲突）

```yaml
# 配置文件1: config_port8402.yaml
api:
  port: 8402

# 配置文件2: config_port8403.yaml
api:
  port: 8403
```

启动多个服务：
```bash
# 终端1
python runapi.py  # 默认使用 config.yaml (8402)

# 终端2
cp config.yaml config_custom.yaml
# 编辑 config_custom.yaml 修改端口为 8403
python runapi.py  # 需要修改代码支持自定义config路径
```

## 监控和调试

### 查看服务状态
```bash
curl http://localhost:8402/health
```

返回：
```json
{
  "status": "healthy",
  "models_loaded": true,
  "gpu_devices": "0,1,2,3,4,5",
  "config_path": "/path/to/config.yaml"
}
```

### 查看配置信息
```bash
curl http://localhost:8402/
```

返回：
```json
{
  "status": "running",
  "message": "DExperts API Server",
  "config": {
    "models": {
      "base": "/storage/data/original_models/Qwen2.5-32B",
      "expert": "/storage/data/original_models/Qwen3-8B",
      "antiexpert": "/storage/data/original_models/Qwen3-8B-base"
    },
    "gpu_devices": "0,1,2,3,4,5"
  }
}
```

### GPU使用情况监控
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者使用 gpustat
pip install gpustat
watch -n 1 gpustat
```

## 常见问题

### Q1: 如何修改模型路径？
A: 编辑 `config.yaml` 中的 `models.*.path`

### Q2: 如何更改端口？
A: 编辑 `config.yaml` 中的 `api.port`

### Q3: GPU内存不足怎么办？
A:
1. 减少模型数量（只用1-2个）
2. 使用更小的模型
3. 在config中添加8bit量化：
```yaml
advanced:
  load_in_8bit: true
```

### Q4: 如何查看详细日志？
A: 在runapi.py顶部修改日志级别：
```python
logging.basicConfig(level=logging.DEBUG)  # 改为DEBUG
```

### Q5: 如何调整DExperts的混合强度？
A: 修改 `generation.alpha` 参数：
- `alpha=0`: 只用base模型
- `alpha=1.0`: 标准DExperts
- `alpha=2.0`: 更强的expert影响
