import json
import argparse
from tqdm import tqdm
from pathlib import Path
# from datasets import load_dataset
# from evaluate import load
import statistics
import json
from collections import defaultdict
import os
import evaluate
from ipdb import set_trace as bp
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# evaluate fackKB: Put your huggingface access tokens
access_token = 
tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2, use_auth_token=access_token)

def evaluate_qa(index2ex, eval_file):
    print(eval_file)
    all_gold = []
    all_pred = []
    all_doc = []
    all_fact_score = []

    if os.path.exists(eval_file) == False:
        return 0
    with open(eval_file, "r") as f:
        output_data = [json.loads(line) for line in f]
    cov_em_all = []
    category2em = defaultdict(list)
    id2ex_output = {}
    for i, output in enumerate(output_data):
        index = output["input_index"]
        pred = output["string"][0]
        gold = index2ex[index]["gold_answers"] 
        if len(pred) < 3:
            print(pred)
            continue
        all_gold.append(gold)
        all_pred.append(pred)
        if len(pred) < 3:
            print(f"pred: {pred}")

        article = index2ex[index]["article"]
        summary = pred
        input = [[summary, article]]
        tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True)
        result = torch.softmax(factkb(**tokens).logits, dim = 1)
        # bp()
        fact_score = result[0][1].item()

        all_fact_score.append(fact_score)
        all_doc.append(article)
        output_dict = index2ex[index].copy()
        output_dict["pred"] = pred
        id2ex_output[i] = output_dict

    print("fact_score: ", statistics.mean(all_fact_score))
    # print(statistics.mean(cov_em_all))
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_pred, references=all_gold)
    print("rouge results: ", results)

    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
    # print("bertscore: ", results)
    print("bertscore: ")
    for k, v in results.items():
        if k in ["precision", "recall", "f1"]:
            print(f"{k}: {statistics.mean(v)}")
    return id2ex_output

# read data
def entity_data(dataset_path):
    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["assigned_process"] == 0:
                raw_data.append(ex)
            # break
        # raw_data = json.loads(f.read())
    return raw_data


if __name__ == "__main__":
    # args parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./eval/cnndm_example_input/cnndm_1_0.jsonl")
    parser.add_argument("--pred_path", type=str, default="./eval/cnndm_example_input/cnndm_1.5_-0.5.jsonl.output_topp0.9_genlen100.jsonl")
    args = parser.parse_args()

    data_path = args.data_path
    pred_path = args.pred_path
    index2ex = entity_data(data_path)
    evaluate_qa(index2ex, pred_path)
    
    

