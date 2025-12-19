import sglang as sgl
import torch

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from sglang_dexpert import DExpertsMultiGPU

def load_dexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: str = None,
    alpha: float = 1.0,
):
    from transformers import AutoTokenizer
    tokenizer_base = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
)   
    tokenizer_expert = AutoTokenizer.from_pretrained(
        expert_model_name_or_path)
    model = DExpertsMultiGPU(
        base_model_path=base_model_name_or_path,
        expert_model_path=expert_model_name_or_path,
        anti_model_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer_base,
        alpha=alpha)
    return model,tokenizer_base,tokenizer_expert

def generate_completions(
    model,
    prompts_an,
    temperature = 0.6,
    top_k=20,
    top_p=0.95,
    max_new_tokens=16000,
    tokenizer_expert=None,
    tokenizer_base = None
):
    sampling_params  ={
        "max_new_tokens": 1,
        "top_k": top_k,
        "temperature":temperature,
        "top_p":top_p
    }
    output =  model.generate(max_new_tokens=max_new_tokens,sampling_params=sampling_params,prompt=prompts_an,tokenizer=tokenizer_base)



    return output 




