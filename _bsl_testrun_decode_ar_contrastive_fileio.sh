#!/usr/bin/bash
trap "kill 0" EXIT

numgpu=2 # should correspond to number of processes in the input jsonl file
available_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

script_role="host" 
global_seed=$1 # inline param, 2021, 2022, etc
single_device_cuda="0" # inline param, "0", "1", etc
multi_device_cuda=$2 # inline param, "0,1,2,3", "0", etc
hf_cache="/private/home/xhan77/.cache/huggingface" 
core_lm_name="specify_in_input_jsonl|n/a" # facebook/opt-1.3b, google/flan-t5-xl
main_log_dir="/private/home/xhan77/cad-private/bsl_logging"

interpret_dataset_tokenized_path="skip"

# data hyperparameters
global_max_seq_len=2048 # 200, 500
####

# retrain
retrain_num_train_epochs=10000
retrain_per_device_train_batch_size=1 
retrain_per_device_eval_batch_size=1
retrain_learning_rate=$3
retrain_weight_decay=0.01
retrain_gradient_accumulation_steps=1
retrain_num_warmup_steps=2000
retrain_max_train_steps=100000

sigma_num_steps=$4
loss_mode=$5
remove_noise_mode=$6
pa=$7
cs=1 # placeholder
dbs=$8
noise_manual_scale=$9
subdir=${10}
decode_context_size=${11}
decode_truncate_len=${12}
decode_depth=${13}
decode_ctr_lr=${14}
projection_top_p=${15}
out_fn=${16}

################ START ################

CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate launch \
    --multi_gpu --mixed_precision fp16 \
    --num_processes ${numgpu} --num_machines 1 --machine_rank 0 \
    --main_process_port ${available_port} \
    --num_cpu_threads_per_process 4 \
    bsl_ar_contrastive_decode_fileio.py \
    --max_seq_length ${global_max_seq_len} \
    --model_name_or_path ${core_lm_name} \
    --num_train_epochs ${retrain_num_train_epochs} \
    --per_device_train_batch_size ${retrain_per_device_train_batch_size} \
    --per_device_eval_batch_size ${retrain_per_device_eval_batch_size} \
    --learning_rate ${retrain_learning_rate} \
    --weight_decay ${retrain_weight_decay} \
    --gradient_accumulation_steps ${retrain_gradient_accumulation_steps} \
    --num_warmup_steps ${retrain_num_warmup_steps} \
    --max_train_steps ${retrain_max_train_steps} \
    --seed ${global_seed} \
    --use_slow_tokenizer \
    --output_dir ${main_log_dir}/${subdir} \
    --loss_mode ${loss_mode} \
    --remove_noise_mode ${remove_noise_mode} \
    --hardcoded_pseudo_diralpha ${pa} \
    --context_size ${cs} \
    --decoding_block_size ${dbs} \
    --sigma_num_steps ${sigma_num_steps} \
    --tokenized_data_file_path ${interpret_dataset_tokenized_path} \
    --if_create_tokenized_data_file "no" \
    --decode_context_size ${decode_context_size} \
    --decode_truncate_len ${decode_truncate_len} \
    --decode_depth ${decode_depth} \
    --train_mode decode \
    --decode_ctr_lr ${decode_ctr_lr} \
    --projection_top_p ${projection_top_p} \
    --out_fn ${out_fn} \
    --decode_mode sim_ctr # decode_mode does not matter as long as ctr_lr is 0
