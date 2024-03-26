#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os

import datasets
import torch

import transformers
import accelerate
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
)
from transformers.utils.versions import require_version

import numpy as np
from termcolor import colored
import json
from accelerate import InitProcessGroupKwargs
import datetime


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, torch.finfo(logits.dtype).min)
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return (2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value) #.to(logits.dtype)


def filter_logits_top_p(logits, top_p, negative_multiplier=False):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    if negative_multiplier:
        filtered_logits = logits.masked_fill(valid_indices == 0, 1000)
    else:
        filtered_logits = logits.masked_fill(valid_indices == 0, -1000)
    return filtered_logits


def decode(args, batch_input_ids, dec_depth, total_t, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size
    
    # for each decode step
    assert args.decode_truncate_len >= 0
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0 # assuming decoding lengths are evenly distributed
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        raise ValueError("context cannot be none for now, change later")
        unit_context_input_ids = None
    history_decode_ids = None

    past_key_values = None # necessary for causal models
    if args.model_category == 'seq2seq':
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            batch_input_ids[:, :args.context_size].clone(), dict(), None
        ) # this step includes encoding the context
        history_decode_ids = model._prepare_decoder_input_ids_for_generation(
            batch_input_ids.size(0),
            model_kwargs=model_kwargs,
            device=batch_input_ids.device,
        ) # create placeholder starter seq for decoding
    else:
        model_kwargs = None

    for i in range(dec_depth):
        # args.accelerator.wait_for_everyone()

        if args.model_category == 'causal':
            model_inputs = model.prepare_inputs_for_generation(unit_context_input_ids, past_key_values=past_key_values)
            outputs = model(**model_inputs, output_hidden_states=False)
        elif args.model_category == 'seq2seq':
            # if history_decode_ids is None:
            #     to_be_shifted = torch.zeros_like(batch_input_ids[:, :1]) # just a placeholder
            # else:
            #     to_be_shifted = torch.cat((history_decode_ids, torch.zeros_like(batch_input_ids[:, :1])), dim=1)
            # model._shift_right(to_be_shifted)

            model_inputs = model.prepare_inputs_for_generation(history_decode_ids, **model_kwargs) # this incorporates past_key_values
            outputs = model(**model_inputs, output_hidden_states=False)

            # outputs = model(input_ids=batch_input_ids[:, :args.context_size].clone(), decoder_input_ids=model._shift_right(to_be_shifted), output_hidden_states=False)
        else:
            raise ValueError("model category not supported")

        equivalent_score = outputs.logits[:, -1:, :].clone().contiguous()

        if args.assigned_weight >= 0:
            equivalent_score = filter_logits_top_p(equivalent_score, top_p=args.filter_top_p)
        else:
            equivalent_score = filter_logits_top_p(equivalent_score, top_p=args.filter_top_p_prior, negative_multiplier=True)

        equivalent_score = args.assigned_weight * equivalent_score
        # equivalent_score = args.accelerator.gather(equivalent_score).sum(dim=0, keepdim=True)
        torch.distributed.all_reduce(equivalent_score, group=args.gathering_group)

        projected_logits = logits_sampling_projection(equivalent_score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)

        if not args.accelerator.is_main_process:
            projected_logits = torch.zeros_like(projected_logits)
        # projected_logits = args.accelerator.reduce(projected_logits, reduction="sum")
        torch.distributed.all_reduce(projected_logits, group=args.gathering_group)

        simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
        # sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
        # logger.info(f"{colored(str(sampled_sequences), 'red')}")

        if args.model_category == 'causal':
            unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1) # not really necessary but keeping

        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.model_category == 'causal':
            past_key_values = outputs.past_key_values
        elif args.model_category == 'seq2seq':
            model_kwargs["past_key_values"] = outputs.past_key_values
            # model_kwargs = model._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            # )

        # HACK: stop when seeing eos token, but asserting batch size is 1, unit_seq_len is 1
        assert real_token_ids_list.size(0) == 1
        assert real_token_ids_list.size(1) == 1
        if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
            break

    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        if args.reverse_lm:
            context_sequences = tokenizer.batch_decode(torch.fliplr(init_context_input_ids).detach().to('cpu'))#, skip_special_tokens=True)
        else:
            context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))#, skip_special_tokens=True)
    else:
        init_context_input_ids = None
        context_sequences = None
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
    logger.info(f"context: {context_sequences}")
    # logger.info(f"gold: {colored(str(gold_sequences), 'yellow')}")
    logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")

    # # debug
    # gen = model.generate(inputs=batch_input_ids[:, :args.context_size], num_beams=1, do_sample=False, max_new_tokens=dec_depth)
    # if args.accelerator.is_main_process:
    #     print(args.tokenizer.batch_decode(gen)) # confirmed our generation has a same effect as model.generate()
    #     breakpoint()
    # args.accelerator.wait_for_everyone()

    return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    # Han: many arguments below will not be used, but keeping for future edits
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library). For example, Wikipedia.",
    )
    parser.add_argument(
        "--additional_dataset_name",
        type=str,
        default=None,
        help="The name of the additional dataset to use (via the datasets library). For example, BookCorpus.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--raw_data_percentage",
        default=100,
        help="The percentage of raw data used as the train set",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.0, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--no_save_grads", action="store_true", help="Whether to save gradients to a file.")
    # for computing influence scores w.r.t. the querying file
    parser.add_argument(
        "--query_file", type=str, default=None, help="A pickle file containing gradient information from the querying data."
    )
    parser.add_argument(
        "--query_data_cap", type=int, default=None, help="Max number of data for which we will save gradients.",
    )
    parser.add_argument("--influence_metric", type=str, default=None, help="Metric for computing the gradients.")
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--tokenized_data_file_path", type=str, default=None, help="Path of the tokenized data file."
    )
    parser.add_argument(
        "--if_create_tokenized_data_file", type=str, default=None, help="Whether to create a new tokenized data file (yes or no)."
    )
    parser.add_argument(
        "--sigma_start_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_end_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_num_steps", type=int, default=1000, help="",
    )
    parser.add_argument(
        "--loss_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--remove_noise_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--hardcoded_pseudo_diralpha", type=float, default=3, help="",
    )
    parser.add_argument(
        "--context_size", type=int, default=0, help="",
    )
    parser.add_argument(
        "--decoding_block_size", type=int, default=25, help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--noise_manual_scale", type=float, default=1, help="",
    )
    parser.add_argument(
        "--decode_context_size", type=int, default=25, help="",
    ) # how many to cut from left
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) # how many to cut from right
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--decode_ctr_lr", type=float, default=0.0, help="",
    )
    parser.add_argument(
        "--out_fn", type=str, default="_sample_gen.jsonl", help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument("--reverse_lm", action="store_true", help="Whether modeling from right-to-left.")
    parser.add_argument(
        "--decode_mode", type=str, default="no_ctr", help="",
    )
    parser.add_argument(
        "--filter_top_p", type=float, default=1.0, help="",
    )
    parser.add_argument(
        "--filter_top_p_prior", type=float, default=1.0, help="",
    )
    parser.add_argument("--big_model_inference", type=str, default="no")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=259200))])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        # set_seed(args.seed)
        accelerate.utils.set_seed(args.seed, device_specific=True) # differ slightly for each device

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # under decode mode, will load model from the output_dir
    if args.train_mode == "decode":
        if len(args.model_name_or_path.split('|')) > 0:
            main_model_name = args.model_name_or_path.split('|')[0]
            fallback_model_name = args.model_name_or_path.split('|')[1]
            args.model_name_or_path = main_model_name
            args.orig_model_name_or_path = fallback_model_name
        else:
            args.orig_model_name_or_path = args.model_name_or_path
    elif args.train_mode == "train":
        raise ValueError("training should be in a separate file (irrelevant in CAD)")

    # Han: assign ensemble models
    args.remove_noise_mode = args.remove_noise_mode.split('|')
    assert args.remove_noise_mode[0] == "fin"
    assert os.path.exists(args.remove_noise_mode[1])
    fin_path = args.remove_noise_mode[1]
    fin_data = []
    with open(fin_path, 'r', encoding='utf-8') as f:
        for line in f:
            # proc_line = line.split('//')[0].strip() # Han: remove debugging mode
            proc_line = line.strip()
            if proc_line:
                fin_data.append(json.loads(proc_line))
    rank2model = dict()
    for _fd in fin_data:
        if _fd['assigned_process'] in rank2model: # sanity check
            assert ' '.join(rank2model[_fd['assigned_process']]) == ' '.join(_fd['assigned_model'].split('|'))
        else:
            rank2model[_fd['assigned_process']] = _fd['assigned_model'].split('|') # first model_path, then suffix like "reverse"

    # Han: add gathering group
    default_backend = torch.distributed.get_backend(torch.distributed.distributed_c10d._get_default_group())
    args.gathering_group = torch.distributed.new_group(ranks=list(sorted(rank2model.keys())), backend=default_backend)

    if accelerator.process_index not in rank2model.keys(): # Han: exit if not in the ensemble
        return
    args.reverse_lm = False
    args.model_name_or_path = rank2model[accelerator.process_index][0]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        if 'llama' in args.model_name_or_path.lower():
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(args.model_name_or_path)
        else:
            try:
                config = AutoConfig.from_pretrained(args.model_name_or_path)
            except:
                raise ValueError("disabled for baseline model")
                # (HACK) change later to make training procedure save the config file
                config = AutoConfig.from_pretrained(args.orig_model_name_or_path)
                config.vocab_size = 50265 # we resized the default OPT model with the default OPT tokenizer
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if 'neox' in args.model_name_or_path.lower(): # Han: gpt-neox doesn't have a slow tokenizer, use GPTNeoXTokenizerFast
        from transformers import GPTNeoXTokenizerFast
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.model_name_or_path)
    elif 'llama' in args.model_name_or_path.lower():
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        assert args.use_slow_tokenizer == True 
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

    if args.init_blank_language_model:
        raise ValueError("disabled for now")
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_name_or_path:
        # Han: modified back for AR compatibility
        if 't5' in args.model_name_or_path.lower() or 'tk' in args.model_name_or_path.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                ignore_mismatched_sizes=False,
                torch_dtype=torch.float16, # Han: this can save some GPU memory
            )
            args.model_category = 'seq2seq'
            model = model.to(accelerator.device)
        else:
            if 'llama' in args.model_name_or_path.lower(): # llama special case
                from transformers import LlamaForCausalLM
                if args.big_model_inference == 'no':
                    model = LlamaForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        torch_dtype=torch.float16, # Han: this can save some GPU memory
                    )
                    model = model.to(accelerator.device)
                else:
                    # Han: we assume 8 GPUs
                    if accelerator.process_index == 0:
                        local_devices = [0, 2, 4, 6]
                    elif accelerator.process_index == 1:
                        local_devices = [1, 3, 5, 7]
                    else:
                        raise ValueError("check accelerator.process_index")
                    # currently, its OPT-structure-specific
                    my_device_map = {'model.embed_tokens': local_devices[0],
                                    'lm_head': local_devices[0],
                                    # 'model.decoder.embed_positions': local_devices[0], # Han: Llama's positional embedding is inside the self attention layer
                                    # 'model.decoder.embedding_sum_layer': local_devices[0],
                                    # 'model.decoder.timestep_layer': local_devices[0],
                                    # 'model.decoder.context_timestep_layer': local_devices[0],
                                    'model.norm': local_devices[0]}
                    for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                        for layer_idx in layer_idx_list:
                            my_device_map[f'model.layers.{layer_idx}'] = local_devices[_device_i]
                    model = LlamaForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        device_map=my_device_map,
                        torch_dtype=torch.float16,
                    )
            elif args.big_model_inference == 'no':
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    torch_dtype=torch.float16, # Han: this can save some GPU memory
                )
                model = model.to(accelerator.device)
            elif args.big_model_inference == 'yes' and 'opt' in args.model_name_or_path.lower():
                # Han: we assume 8 GPUs
                if accelerator.process_index == 0:
                    local_devices = [0, 2, 4, 6]
                elif accelerator.process_index == 1:
                    local_devices = [1, 3, 5, 7]
                else:
                    raise ValueError("check accelerator.process_index")
                # currently, its OPT-structure-specific
                my_device_map = {'model.decoder.embed_tokens': local_devices[0],
                                'lm_head': local_devices[0],
                                'model.decoder.embed_positions': local_devices[0],
                                # 'model.decoder.embedding_sum_layer': local_devices[0],
                                # 'model.decoder.timestep_layer': local_devices[0],
                                # 'model.decoder.context_timestep_layer': local_devices[0],
                                'model.decoder.final_layer_norm': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'model.decoder.layers.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            elif args.big_model_inference == 'yes' and 'neox' in args.model_name_or_path.lower():
                # Han: we assume 8 GPUs
                if accelerator.process_index == 0:
                    local_devices = [0, 2, 4, 6]
                elif accelerator.process_index == 1:
                    local_devices = [1, 3, 5, 7]
                else:
                    raise ValueError("check accelerator.process_index")
                # currently, its Neo-structure-specific
                my_device_map = {'gpt_neox.embed_in': local_devices[0],
                                'embed_out': local_devices[0],
                                'gpt_neox.final_layer_norm': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'gpt_neox.layers.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            elif args.big_model_inference == 'yes' and 'neo' in args.model_name_or_path.lower():
                # Han: we assume 8 GPUs
                if accelerator.process_index == 0:
                    local_devices = [0, 2, 4, 6]
                elif accelerator.process_index == 1:
                    local_devices = [1, 3, 5, 7]
                else:
                    raise ValueError("check accelerator.process_index")
                # currently, its Neo-structure-specific
                my_device_map = {'transformer.wte': local_devices[0],
                                'lm_head': local_devices[0],
                                'transformer.wpe': local_devices[0],
                                'transformer.drop': local_devices[0],
                                'transformer.ln_f': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'transformer.h.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            else:
                raise ValueError("check args.big_model_inference")

            args.model_category = 'causal'
        model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward) # referred to https://github.com/huggingface/accelerate/blob/38fd30e764ea87ef86e7d69fcba559c3605925b1/src/accelerate/accelerator.py#L1138
        model.forward = accelerate.utils.convert_outputs_to_fp32(model.forward)
    else:
        raise ValueError("specify --init_blank_language_model")
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"model size: {sum(p.numel() for p in model.parameters())}")
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)

    # Save accelerator state
    if args.train_mode == "resume": # resuming job would still break the strict reproducibility, since we are not saving noise states
        raise ValueError("use the decode mode")
        accelerator.load_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        with open(os.path.join(args.output_dir, "completed_steps.txt"), 'r') as f:
            completed_steps = int(f.read())
    elif args.train_mode == "train":
        raise ValueError("use the decode mode")
        if os.path.exists(os.path.join(args.output_dir, 'accelerate_ckpt')):
            logger.info("training probably interrupted, should change mode to resume for the next run")
            return 0
        accelerator.save_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        completed_steps = 0
    elif args.train_mode == "decode":
        pass
    else:
        raise ValueError("train_mode must be one of 'train', 'resume', 'decode'")

    total_t = args.sigma_num_steps
    one_hot_value = args.hardcoded_pseudo_diralpha # for a pseudo one-hot encoding for alpha

    args.noise_analysis_list = list()

    if args.train_mode == "train" or args.train_mode == "resume":
        raise ValueError("Training or resuming is disabled here")

    ##########################################

    out_json_fn = f"{fin_path}.output_topp{args.projection_top_p}_genlen{args.decode_depth}.jsonl" # os.path.join(args.output_dir, args.out_fn)
    if accelerator.is_main_process:
        with open(out_json_fn, 'w') as f:
            f.write('placeholder, program not finished\n')

    args.tokenizer = tokenizer

    # Decoding, includes hardcode for now
    if args.train_mode == "decode":
        model.eval()

        args.context_size = args.decode_context_size
        args.one_hot_value = one_hot_value
        args.vocab_size = vocab_size
        args.hidden_size = hidden_size
        args.accelerator = accelerator
        args.ctr_model = None

        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len
        with torch.no_grad():
            for step, _fd in enumerate(fin_data): # only support batch size 1 since the context size can be different across lines
                if _fd['assigned_process'] != args.accelerator.process_index: # remember to unblock barriers before this line
                    continue
                args.assigned_weight = _fd['assigned_weight']

                ctx_field_name = 'context_string'
                assert ctx_field_name in _fd
                assert args.per_device_eval_batch_size == 1

                input_ids = torch.LongTensor(tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True)).unsqueeze(0).to(args.accelerator.device)
                if args.reverse_lm:
                    input_ids = torch.fliplr(input_ids)
                else:
                    pass
                args.context_size = input_ids.size(1)
                args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size # Han: this compensates for the unknown input context size

                if 'filter_p' in _fd: # Han: token filtering
                    args.filter_top_p = _fd['filter_p']
                if 'filter_p_prior' in _fd:
                    args.filter_top_p_prior = _fd['filter_p_prior']

                if args.decode_truncate_len < 0:
                    continue # skipping very long examples
                logger.info(f"idx: {_fd['input_index']}")

                repeat_sample = 1 # currently change here manually
                for _r in range(repeat_sample):
                    history_decode_ids, _, _, sampled_sequences, _, _ = \
                        decode(args, input_ids, args.decode_depth, total_t, model, tokenizer)
                    if _r == 0: # first sample
                        # export to jsonl
                        for _i in range(args.per_device_eval_batch_size):
                            export_dict = dict()
                            export_dict['tokens'] = [history_decode_ids.tolist()[_i]]
                            export_dict['string'] = [sampled_sequences[_i]]
                            export_dict['assigned_process'] = _fd['assigned_process']
                            export_dict['assigned_model'] = args.model_name_or_path
                            export_dict['output_index'] = len(export_list)
                            export_dict['input_index'] = _fd['input_index']
                            export_list.append(export_dict)
                    else:
                        for _i in range(args.per_device_eval_batch_size):
                            export_list[-(args.per_device_eval_batch_size - _i)]['tokens'].append(history_decode_ids.tolist()[_i])
                            export_list[-(args.per_device_eval_batch_size - _i)]['string'].append(sampled_sequences[_i])

        if accelerator.is_main_process:
            if os.path.exists(out_json_fn):
                os.remove(out_json_fn)
                logger.info(f"Cleaning existing {out_json_fn}")
            with open(out_json_fn, mode="w") as f_out: # use mode a if several processes are wirting to the same file
                for export in export_list:
                    f_out.write(json.dumps(export))
                    f_out.write("\n")


if __name__ == "__main__":
    main()
