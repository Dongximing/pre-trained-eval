from paser import *
import os
import json
from tqdm import tqdm
import transformers
import argparse
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

    parser.add_argument('--eval_path', type=str, default='/home/ximing/pre-trained-eval/proxy-tuning/qwen2.5_14b_result')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()


    number_correct = 0

    total_number = args.end - args.start
    wrong_list =  []



    for idx, number in enumerate(tqdm(range(args.start, args.end))):
      
        
        json_path = os.path.join(args.eval_path, f"data_{idx}.json")
        if not os.path.exists(json_path):
            wrong_list.append(number)
            print(f"[Warning] {json_path} does not exist, skipping...")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            predict = generations[0].get('output')[0]
            standard = generations[0].get('answer')
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
