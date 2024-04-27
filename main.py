import argparse
import os
import datetime
import json
import transformers
from transformers.utils import logging
import torch
import time
import datasets


API_TOKEN_COUNTER = 0
DEVICE = "cuda"

def create_save_folder(args, start_time):
    base_model_name = args.base_model.replace('/', '_')
    save_folder_name = f"results/{args.output_name}{base_model_name}-{args.mask_filling_model}/{start_time}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    logger.info(f"Saving results to path: {os.path.abspath(save_folder_name)}")
    return save_folder_name

def setup_cache(args):
    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    logger.info(f"Cache directory: {args.cache_dir}")

def get_base_model(model_name, cache_dir):
    # no openai model not chosen
    logger.info(f'Loading Base model: {model_name}...')
    model_kwargs = {}
    if 'gpt-j' in model_name or 'neox' in model_name:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def load_base_model():
    logger.info("Moving Base Model to GPU...", end='', flush=True)
    start = time.time()
    base_model.to(DEVICE)
    logger.info(f'Done ({time.time() - start:.2f}s)')



if __name__ == '__main__':
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--pct-words-masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span-length', type=int, default=2)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--n-perturbation-list', type=str, default="1,10")
    parser.add_argument('--base-model', type=str, default="gpt2-medium")
    parser.add_argument('--mask-filling-model', type=str, default="t5-large")
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=20)
    parser.add_argument('--cache-dir', type=str, default=".cache")
    parser.add_argument('--output-name', type=str, default="n_perturb")
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--random-fills', action='store_true')
    args = parser.parse_args()

    logger.info(args.dataset)
    logger.info(args.mask_filling_model)
    logger.info(args.n_perturbation_list)

    # torch.cuda.empty_cache()

    # considering int8 precision 
    # considering top-k sampling for text generation

    formatted_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # save folder for results - 
    save_folder_name = create_save_folder(args, formatted_start_time)

    with open(os.path.join(save_folder_name, "arguments.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    setup_cache(args)
    perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    base_model, base_tokenizer = get_base_model(args.base_model, args.cache_dir)
    
    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)

    # considering only baselines and random fills
    #considering int8
    kwargs = dict(load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16)
    logger.info(f'Loading mask filling model {args.mask_filling_model}')
    
    # use sequence-to-sequence LM since we want to map the input text to a masked output text
    # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model, **kwargs, cache_dir=args.cache_dir)
    
    pre_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=256, cache_dir=args.cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model, model_max_length=256, cache_dir=args.cache_dir)

    load_base_model(base_model)

    

    
