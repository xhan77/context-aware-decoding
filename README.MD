[Very Raw]

## Overview
Our work explores the setup where two identical language models are decoded contrastively with different input contexts. However, our code generally supports collaborative or contrastive decoding with multiple language models with different input contexts, by customizing your own input `jsonl` file for inference. 

## Input format
```json
{
    "input_index": 0, 
    "assigned_model": "huggyllama/llama-7b", 
    "assigned_process": 0, 
    "context_string": "The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 1078 episodes . How many episodes are in chicago fire season 4 ?", 
    "assigned_weight": 2,
    "filter_p": 1.0,
}
{
    "input_index": 0,
    "assigned_model": "huggyllama/llama-7b",
    "assigned_process": 1,
    "context_string": "How many episodes are in chicago fire season 4 ?", 
    "assigned_weight": -1,
}
...
```

## Context-aware decoding on CNN-DM and NQ-Swap data
Run `bash exp_cnndm.sh` or `bash exp_nqswap.sh`. Both scripts call `run_group_decode_fileio.sh` which calls `group_decode_fileio.py`.

### Evaluation
After generating the prediction data, you can run the evaluation by running the following script and compare with the gold data.  
```bash
PRED_PATH=$YOUR_PREDICTION_PATH
GOLD_DATA_PATH=/fsx-onellm/swj0419/cad/cad-private/eval/gold/likert_evaluation_results.json
python eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH
```
