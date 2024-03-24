
### Evaluation
After generating the prediction data, you can run the evaluation by running the following script and compare with the gold data.  
```bash
PRED_PATH=$YOUR_PREDICTION_PATH
GOLD_DATA_PATH=/fsx-onellm/swj0419/cad/cad-private/eval/gold/likert_evaluation_results.json
python eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH
```
