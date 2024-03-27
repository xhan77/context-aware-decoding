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

# evaluate fackKB
access_token = "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"
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
    with open(dataset_path) as f:
        raw_data = json.loads(f.read())

    prompt = ""
    count = 0
    index2ex = {}
    data = []
    # read demos
    all_articles = []

    for i, d in enumerate(raw_data):
        ex = d
        if ex["model"] == "openai_text-davinci-002 (0 shot)" and ex["dataset"] == "xsum":
            gold_label = ex["summary"]
            # if ex['article'][:50] not in article2gold_sum:
            #     continue
            # gold_label = article2gold_sum[ex['article'][:50]]

            article = ex['article'].replace("\n", " ")
            if len(article.split()) > 800:
                continue
            if article[:50] in all_articles:
                # print("duplicate")
                continue
            ex_text = prompt + f"Article: {article} Summarize the article in one sentence. Summary:"  
            ex_text_query_only = f"Summarize the article in one sentence. Summary:" 


            new_ex = {}
            new_ex[f"input_index"] = count
            new_ex["assigned_process"] = 0

            # swj change
            new_ex["context_string"] = ex_text
            new_ex["gold_answers"] = gold_label
            data.append(new_ex)
            new_ex["article"] = article
            index2ex[count] = new_ex
            
            new_ex = {}
            new_ex[f"input_index"] = count
            new_ex["assigned_process"] = 1
            new_ex["context_string"] =  ex_text_query_only  
            new_ex["gold_answers"] = gold_label
            count += 1
            data.append(new_ex)
    return index2ex, data


if __name__ == "__main__":
    # args parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./eval/gold/likert_evaluation_results.json")
    parser.add_argument("--pred_path", type=str, default="./eval/cnndm_example_input/cnndm_1_0.jsonl")
    args = parser.parse_args()

    data_path = args.data_path
    pred_path = args.pred_path
    index2ex, data = entity_data(data_path)
    evaluate_qa(index2ex, pred_path)
    
    

