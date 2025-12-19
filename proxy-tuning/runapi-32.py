# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List

# import torch

# # === 1. import your existing code ===
# from generation import generate_completions, load_dexperts_model_and_tokenizer
# # 或 from your file that loads normal model
# # from somefile import load_lm_and_tokenizer
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
# app = FastAPI()

# print("🚀 Loading models ONCE on GPU ...")

# # === 2. Load model ONCE when the server starts ===
# model, tokenizer = load_dexperts_model_and_tokenizer(
#     base_model_name_or_path="/home/original_models/Qwen1.5-14B",
#     expert_model_name_or_path="/home/original_models/Qwen1.5-7B-Chat",
#     antiexpert_model_name_or_path="/home/original_models/Qwen1.5-7B",
#     alpha=1.0,
# )

# # model.eval()  # 不训练，只推理
# # print("🔥 Models loaded and ready!")


# # === 3. Request body ===
# class GenRequest(BaseModel):
#     prompts: List[str]
#     answers: List[str] = []  # 可选，不要求你改逻辑
#     max_new_tokens: int = 160000
#     temperature: float = 1.0
#     top_p: float = 1.0


# # === 4. API endpoint ===
# @app.post("/generate")
# def generate(req: GenRequest):
#     """
#     调用你原来的 generation.py
#     没有改你的生成逻辑！
#     """
#     print(req.prompts)
#     results = generate_completions(
#         model=model,
#         tokenizer=tokenizer,
#         prompts_an=(req.prompts, req.answers),
#         batch_size=4,
#         max_new_tokens=req.max_new_tokens,
#         temperature=req.temperature,
#         top_p=req.top_p,
#         save_dir=None,        
#         disable_tqdm=True
#     )

#     return {"results": results}


# # === 5. Entry ===
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8888)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import torch

from generation_qwen3 import generate_completions, load_dexperts_model_and_tokenizer



app = FastAPI()

print("🚀 Loading DExperts...")
model, tokenizer = load_dexperts_model_and_tokenizer(
    base_model_name_or_path="/home/original_models/Qwen3-8B-base",
    expert_model_name_or_path="/home/original_models/Qwen3-4B",
    antiexpert_model_name_or_path="/home/original_models/Qwen3-4B-base",
    alpha=1,
)
print("🔥 DExperts loaded!")


# =========================
# OpenAI Chat Completion API
# =========================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "dexperts"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 16000
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0


@app.post("/v1/chat/completions")
def chat_completion(req: ChatCompletionRequest):

    # 1) 将 messages 拼接成 prompt
    prompt = ""
    for msg in req.messages:
        prompt = msg.content.replace("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n### Question:","") # 生成位置


 

    # 2) 用你已有的 generate_completions 推理
    print(prompt)
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts_an=([prompt], [""]),
        batch_size=1,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        disable_tqdm=True
    )

    text = outputs[0]["output"][0]

    # 3) 构造 OpenAI chat 格式响应
    return {
        "id": "chatcmpl-local-001",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8882)
