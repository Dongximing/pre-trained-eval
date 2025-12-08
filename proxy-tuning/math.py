import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from datasets import load_dataset
from generation import (
    ensure_dir,
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
)


def create_prompt(row):
    return f'Question: {row["question"]}\nAnswer:'


def main(args):
    ensure_dir(args.save_dir)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            args.antiexpert_model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

    ds = load_dataset("HuggingFaceH4/MATH-500")['test']


    ds = ds.select(range(0, 1))
    prompts = []
    problems_and_answers = [item["problem"] for item in ds]
    print(problems_and_answers[0])
    print('--------\n')
   

    outputs = generate_completions(
        model,
        tokenizer,
        problems_and_answers,
        batch_size=args.eval_batch_size,
        do_sample=False,
        max_new_tokens=400,
    )

    test_df['output'] = [o.strip() for o in outputs]
    cors = []
    for i, row in test_df.iterrows():
        # ignore casing
        pred = row['output'].lower()
        answers = [a.strip().lower() for a in row['answers']]
        cors.append(pred in answers)

    test_df['correct'] = cors
    acc = np.nanmean(cors)
    print(f"Accuracy: {np.round(acc, 3)}")

    test_df.to_json(os.path.join(args.save_dir, "predictions.jsonl"), lines=True, orient='records')

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fo:
        json.dump({
            "acc": acc,
            "tot": len(test_df)
        }, fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="math-500"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="qwen_dexperts_math-500_results"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="if specified, a maximum of max_examples for evaluation"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='/home/original_models/Qwen2.5-14B',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='/home/original_models/Qwen2.5-7B-Instruct',
    )
    parser.add_argument(
        "--antiexpert_model_name_or_path",
        type=str,
        default='/home/original_models/Qwen2.5-7B',
    )
    args = parser.parse_args()

    main(args)