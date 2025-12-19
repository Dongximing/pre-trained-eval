from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
import os 
import sglang as sgl
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class TrackFullVocabLogits(CustomLogitProcessor):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.last_logits = None

    def __call__(self, logits, custom_param_list):
        import torch
        assert logits.shape[0] == len(custom_param_list)
        print(logits)
        return logits


if __name__ == "__main__":
    llm = sgl.Engine(
        model_path="Qwen/Qwen3-0.6B",
        enable_custom_logit_processor=True,
    )

    prompt = "The capital of France is"
    sampling_params = {
        "temperature": 1.0,
        "max_new_tokens": 1,
        "custom_params": {
        },   # 必须有，才能承载回传字段（空 dict 也行）
    }

    output = llm.generate(
        prompt,
        sampling_params,
        custom_logit_processor=TrackFullVocabLogits().to_str(),
    )

    print(sampling_params)
    import numpy as np
    loaded_logits = np.load("last_logits.npy")
    print("Loaded logits shape:", loaded_logits)
    
    # token_logits = trck_processor.last_logits  # shape: [1, vocab_size]
    # print("Token logits shape:", token_logits.shape)
    print("Generated:", output["text"])

    llm.shutdown()
