# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
MATH_PROMPT = "You are an expert python programmer. you will be given a question (problem specification) and will generate a correct python program that matches the specification and passes all tests."
def load_lm_and_tokenizer(
    model_name_or_path,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        offload_folder="offload_folder",
        offload_state_dict=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    return model, tokenizer


@torch.inference_mode()
def chat_generate(
    model,
    tokenizer,
    prompts_an,
    max_tokens=16000,
):
    # === 拼 prompt（基础版，后面可换 Qwen 官方 template）===
    
    messages = [
    {"role": "user", "content":  prompts_an[0][0]}
]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
         enable_thinking=False,
    )
    print('chat_text',text)
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True
    ).to(model.device)
   

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text)

    return text
