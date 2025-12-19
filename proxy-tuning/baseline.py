
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import torch

from basemodel import chat_generate, load_lm_and_tokenizer



app = FastAPI()

print("🚀 Loading DExperts...")
model, tokenizer = load_lm_and_tokenizer(
    model_name_or_path="/home/original_models/Qwen3-8B",

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
    outputs = chat_generate(
        model=model,
        tokenizer=tokenizer,
        prompts_an=([prompt], [""]),
        max_tokens=20000,

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
    uvicorn.run(app, host="0.0.0.0", port=8121)
