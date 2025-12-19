# server.py
import multiprocessing as mp
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

from sglang_worker import worker_loop  # 确保同目录

app = FastAPI()

parent_conn = None
child_conn = None
worker = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


@app.on_event("startup")
def startup_event():
    global parent_conn, child_conn, worker

    parent_conn, child_conn = mp.Pipe()

    worker = mp.Process(
        target=worker_loop,
        args=(child_conn,),
        daemon=False,   # 可选，但推荐
    )
    worker.start()

    print("✅ sglang worker started")


@app.post("/v1/chat/completions")
def chat_completion(req: ChatRequest):
    prompt = "\n".join(m.content for m in req.messages)

    parent_conn.send((
        prompt,
        {
            "max_new_tokens": 1,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        },
    ))

    text = parent_conn.recv()

    return {
        "choices": [
            {"message": {"role": "assistant", "content": text}}
        ]
    }


if __name__ == "__main__":
    mp.freeze_support()  # ⭐ 必加（spawn 环境）
    uvicorn.run(app, host="0.0.0.0", port=8333, workers=1)
