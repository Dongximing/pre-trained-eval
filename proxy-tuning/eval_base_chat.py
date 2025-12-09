from paser import *
import os
import json
from tqdm import tqdm
import transformers
import argparse
from datasets import load_dataset
def check_math_correctness(ref, generation):
    if not find_box(generation): return False
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)

    parser.add_argument('--eval_path', type=str, default='/home/ximing/pre-trained-eval/proxy-tuning/qwen2.5-14b-chat.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()


    number_correct = 0

    total_number = args.end - args.start
    wrong_list =  []
    dataset = load_dataset("HuggingFaceH4/math-500")["test"]
    answers = []
    for idx in range(args.start, args.end):
        answers.append(dataset[idx]['answer'])

    with open(args.eval_path, "r", encoding="utf-8") as f:
        generations = json.load(f)
        for idx, number in enumerate(tqdm(range(args.start, args.end))):
            predict = generations[idx].get('output')[0]
            standard = answers[idx]
            print('predict:', predict)
            print('standard:', standard)
            result = check_math_correctness(standard, predict)
            if result:
                number_correct += 1
            else:
                wrong_list.append(idx)

       

    print("correct Number of tokens: ", number_correct/number_correct)
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f"wrong_list: {wrong_list}")
