import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_NAME = "Qwen/Qwen2.5-Math-7B"
INST_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"

# 加载模型权重（只 load state_dict 更快，不构建计算图）
base = AutoModelForCausalLM.from_pretrained(BASE_NAME, torch_dtype=torch.float16, device_map="cpu").state_dict()
inst = AutoModelForCausalLM.from_pretrained(INST_NAME, torch_dtype=torch.float16, device_map="cpu").state_dict()

os.makedirs("qwen_diff_vis", exist_ok=True)

def calc_sigma(Wb, Wi):
    return torch.sum(torch.abs(Wb - Wi)) / (torch.sum(torch.abs(Wb)) + torch.sum(torch.abs(Wi)) + 1e-8)

def draw_heatmap(mat, title, save_path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(mat.cpu().float().numpy(), cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

sigma_report = []

for name, Wb in base.items():
    if name not in inst:
        continue
    Wi = inst[name]

    # 跳过 1D 或 embedding/bias 向量（不可视化）
    if Wb.ndim < 2:
        continue

    sigma = calc_sigma(Wb, Wi).item()
    sigma_report.append((name, sigma))

    # 保存权重图
    folder = os.path.join("qwen_diff_vis", name.replace(".", "_"))
    os.makedirs(folder, exist_ok=True)

    draw_heatmap(Wb, "BASE", os.path.join(folder, "BASE.png"))
    draw_heatmap(Wi, "INSTRUCT", os.path.join(folder, "INSTRUCT.png"))
    draw_heatmap(Wb - Wi, f"DIFF σ={sigma:.4f}", os.path.join(folder, f"DIFF_sigma_{sigma:.4f}.png"))

# σ 排序输出
sigma_report.sort(key=lambda x: x[1], reverse=True)
print("\n===== Qwen Layer-wise σ (descending) =====")
for name, s in sigma_report:
    print(f"{name:<70} σ = {s:.6f}")

print("\n🎉 所有层的差异热力图已保存到 ./qwen_diff_vis/")
