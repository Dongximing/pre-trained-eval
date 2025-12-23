
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import torch

from generation import generate_completions, load_dexperts_model_and_tokenizer
MATH_PROMPT = "You are an expert python programmer. you will be given a question (problem specification) and will generate a correct python program that matches the specification and passes all tests."


app = FastAPI()

print("🚀 Loading DExperts...")
model, tokenizer = load_dexperts_model_and_tokenizer(
    base_model_name_or_path="/home/original_models/Qwen3-30B-A3B-Base",
    expert_model_name_or_path="/home/original_models/Qwen3-14B",
    antiexpert_model_name_or_path="/home/original_models/Qwen3-14B-base",
    alpha=1.0,
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
    prompts =[]
    for msg in req.messages:
        prompt = msg.content
        prompts.append(MATH_PROMPT+prompt)


 

    # 2) 用你已有的 generate_completions 推理
    print(prompts)
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts_an=(prompts, [""]),
        batch_size=2,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        disable_tqdm=True
    )

 

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
                    "content": outputs
                },
                "finish_reason": "stop"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8408)
