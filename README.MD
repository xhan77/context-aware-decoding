[Very Raw]

## Overview
Our work explores the setup where two identical language models are decoded contrastively with different input contexts. However, our code generally supports collaborative or contrastive decoding with multiple language models with different input contexts, by customizing your own input `jsonl` file for inference. 

## Context-aware decoding on CNN-DM and NQ-Swap data
`bash run_cnndm_custom.sh` or `bash run_nqswap_custom.sh`

### Evaluation
After generating the prediction data, you can run the evaluation by running the following script and compare with the gold data.  
```bash
PRED_PATH=$YOUR_PREDICTION_PATH
GOLD_DATA_PATH=/fsx-onellm/swj0419/cad/cad-private/eval/gold/likert_evaluation_results.json
python eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH
```
