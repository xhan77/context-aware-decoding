#!/usr/bin/bash
# trap "kill 0" EXIT

hf_cache="/private/home/xhan77/.cache/huggingface" # CHANGE THIS TO YOUR OWN CACHE PATH

numgpu=2 # should match the number of processes in the input jsonl file, default to 2 for context-aware decoding
available_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
global_seed=$1 
multi_device_cuda=$2 # "0,1,2,3", "0", etc
core_lm_name="specify_in_input_jsonl|n/a" # facebook/opt-1.3b, google/flan-t5-xl, etc.

file_mode=$3
global_max_seq_len=$4 # should be consistent with (<=) the specified models' max_seq_len
decode_truncate_len=$5 # effective gen len is global_max_seq_len - decode_truncate_len
decode_depth=$6
projection_top_p=$7

################ START ################

CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate launch \
    --multi_gpu --mixed_precision fp16 \
    --num_processes ${numgpu} --num_machines 1 --machine_rank 0 \
    --main_process_port ${available_port} \
    --num_cpu_threads_per_process 10 \
    group_decode_fileio.py \
    --max_seq_length ${global_max_seq_len} \
    --model_name_or_path ${core_lm_name} \
    --seed ${global_seed} \
    --use_slow_tokenizer \
    --file_mode ${file_mode} \
    --decode_truncate_len ${decode_truncate_len} \
    --decode_depth ${decode_depth} \
    --train_mode decode \
    --projection_top_p ${projection_top_p} 
