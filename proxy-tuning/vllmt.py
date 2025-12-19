import sglang as sgl
import torch
import multiprocessing as mp
import os 
from transformers import AutoTokenizer
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def main():
    llm = sgl.Engine(model_path="Qwen/Qwen3-0.6B-base")
    messages = [
                # {"role": "system", "content": MATH_PROMPT},
                {"role": "user", "content": "what is the capital of France?"},
        ]
    chat_text = qwen_tok.apply_chat_template(
                messages,
                tokenize=False,        # 先生成字符串

                enable_thinking=False,
            )

    print("Chat text:", chat_text)


    output = llm.generate(
        ["11111"],
        sampling_params={
            "max_new_tokens": 10,
            "top_k": 1,
        },
    )
    print("Generation output:", output[0]["text"])


    logits = output[0]["meta_info"]["output_token_logits"]
    print(f"Generated text: {logits}")
    print(f"logits shape: {logits.shape}")
    print(f"Expected tokens: {torch.argmax(logits, axis=1)}")
    print(f"Actual tokens: {output[0]['output_ids']}")


if __name__ == "__main__":
    mp.freeze_support()   # 必须
    main()
