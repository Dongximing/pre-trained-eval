import os

# (Optional general imports can stay here if they don't spawn processes)
# from transformers import AutoModelForCausalLM, AutoTokenizer  # can also import inside main if desired
# import torch
import torch
import torch.nn as nn

if __name__ == "__main__":
    # If running on Windows or freezing to EXE, uncomment the next two lines:
    # from multiprocessing import freeze_support
    # freeze_support()

    import sglang as sgl
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    model_name = "Qwen/Qwen3-0.6B"  # Model path or name
    llm = sgl.Engine(model_path=model_name, dtype="bfloat16", enable_return_hidden_states=True)  # Initialize SGLang engine (spawns process internally)

    prompt = "hello, my name is"
    sampling_params = {

        "max_new_tokens": 1,
        "top_k": 1,
        "temperature": 1.0,
    }

    outputs = llm.generate([prompt], sampling_params=sampling_params, return_hidden_states=True)
    print("SGLang 输出结果:", outputs[0]["text"])
    for out in outputs:
        logits = out["meta_info"]["output_token_logits"]
        print(f"Generated text: {logits}")
        print(f"logits shape: {logits.shape}")
        print(f"Expected tokens: {torch.argmax(logits, axis=1)}")
        print(f"Actual tokens: {out['output_ids']}")
    

    
    # Load HuggingFace model for comparison
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,trust_remote_code=True).eval()
    hf_model.to("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(hf_model.device)
  
    print("逐时刻比较 logits 差异:")
    tolerance = 1e-4
    all_good = True

    # Get logits from HF model for the prompt (and decode if any)
    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
    print(inputs)
    with torch.no_grad():
        hf_outputs = hf_model(**inputs, output_hidden_states=True,temperature=1.0)
    hf_logits = hf_outputs.logits  # shape: (1, prompt_length, vocab_size)
    print(f"HuggingFace 模型 logits 维度: {hf_logits[:, -1, :]}")
    print(f"Expected tokens: {torch.argmax(hf_logits[:, -1, :], axis=1)}")
    hf_logits_t = hf_logits[:, -1, :]
    diff = logits - hf_logits_t
    mean_diff = diff.abs().mean().item()
    mse_diff = float((diff**2).mean().item())
    print(f"时间步 {1}: 最大绝对差 = {mean_diff:.2e}, MSE = {mse_diff:.2e}")
    if max_diff > tolerance:
        all_good = False

    if all_good:
        print("验证成功：两个来源的 logits 完全一致（差异<1e-4）！")
    else:
        print("验证失败：两个来源的 logits 存在超出阈值的差异！")

    llm.shutdown()  # Properly shut down the SGLang engine to release resources
