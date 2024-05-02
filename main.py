import argparse
import os
import datetime
import json
import transformers
from transformers.utils import logging
import torch
import time
import datasets
import random
import numpy as np


API_TOKEN_COUNTER = 0
DEVICE = "cuda"
MODEL_MAX_LENGTH = 512
MIN_WORDS_SAMPLED = 55
TOP_K = 30


def save_temp_results(data):
    # Open file in write mode
    with open(os.path.join(save_folder_name, "temp_data.json"), "w") as file:
        # Write each item in the dataset to the file
        for item in data:
            file.write(item + "\n")  # Add a newline character after each item

def create_save_folder(args, start_time):
    base_model_name = args.base_model.replace('/', '_')
    save_folder_name = f"results/{args.output_name}{base_model_name}-{args.mask_filling_model}/{start_time}-{args.pct_words_masked}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    logger.warning(f"Saving results to path: {os.path.abspath(save_folder_name)}")
    return save_folder_name

def setup_cache(args):
    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    logger.warning(f"Cache directory: {args.cache_dir}")

def get_base_model(model_name, cache_dir):
    # no openai model not chosen
    logger.warning(f'Loading Base model: {model_name}...')
    model_kwargs = {}
    if 'gpt-j' in model_name or 'neox' in model_name:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def load_base_model(base_model):
    logger.warning("Moving Base Model to GPU...")
    start = time.time()
    base_model.to(DEVICE)
    logger.warning(f'Done ({time.time() - start:.2f}s)')

def trim_to_shortest(text1, text2):
    shorter = min(len(text1.split(' ')), len(text2.split(' ')))
    text1 = " ".join(text1.split(' ')[:shorter])
    text2 = " ".join(text2.split(' ')[:shorter])
    return text1, text2

def sample_from_model(texts, min_words=55, prompt_tokens=30):
    encoded_text = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    encoded_text = {key: value[:, : prompt_tokens] for key,value in encoded_text.items()}
    decoded_text = ['' for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # TODO this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works

    tries = 0
    while(m := min(len(x.split()) for x in decoded_text)) < min_words:
        if tries != 0:
            logger.warn(f"\nmin words: {m}, needed {min_words}, regenerating (try {tries})")
        
        sampling_kwargs = {}
        # for top_k sampling
        sampling_kwargs['top_k'] = TOP_K
        min_length = 150
        outputs = base_model.generate(**encoded_text, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
        decoded_text = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tries += 1

    return decoded_text

def get_samples(data, batch_size):
    # GENERATE SAMPLES
    logger.info(f"Generating samples...")
    torch.manual_seed(31)
    np.random.seed(31)
    data = {
        "original": [],
        "sampled": [],
    }
    for batch in range(len(data) // batch_size):
        logger.info(f"Generating samples for batch {batch} of {len(data) // batch_size}")
        original = data[batch*batch_size: (batch+1)*batch_size]
        sampled = sample_from_model(original, min_words=MIN_WORDS_SAMPLED)

        for o, s in zip(original, sampled):
            o, s = trim_to_shortest(o, s)
            data["original"].append(o)
            data["sampled"].append(s)

    # TODO Introduce pre-perturbations

    return data

        


def get_data(dataset, key, pre_tokenizer, batch_size, cache_dir):
    data = datasets.load_dataset(dataset, split="train", cache_dir=cache_dir)[key]

    # PREPROCESS
    # remove duplicates
    data = list(dict.fromkeys(data))

    # remove new lines
    data = [' '.join(x.split()) for x in data]

    # remove surrounding whitespace
    data = [x.strip() for x in data]

    # Keep sufficiently long examples
    if dataset=='xsum' :
        long_samples = [x for x in data if len(x.split()) > 250]
        if len(long_samples) > 0:
            data = long_samples

    random.seed(1)
    random.shuffle(data)
    data = data[:1000]

    tokenized_data = pre_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= MODEL_MAX_LENGTH]

    logger.warning(f"Total number of samples: {len(data)}")
    logger.warning(f"Avg number of words: {np.mean([len(x.split()) for x in data])}")

    return get_samples(data[:args.n_samples], batch_size=batch_size)




if __name__ == '__main__':
    logging.set_verbosity_warning()
    logging.enable_explicit_format
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

    logger.warning(args.dataset)
    logger.warning(args.mask_filling_model)
    logger.warning(args.n_perturbation_list)

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
    logger.warning(f'Loading mask filling model {args.mask_filling_model}')
    
    # use sequence-to-sequence LM since we want to map the input text to a masked output text
    # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model, **kwargs, cache_dir=args.cache_dir)
    
    pre_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=MODEL_MAX_LENGTH, cache_dir=args.cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model, model_max_length=256, cache_dir=args.cache_dir)

    load_base_model(base_model)

    logger.warning(f'Loading dataset: {args.dataset}...')
    data = get_data(args.dataset,
                    key="document",
                    pre_tokenizer=pre_tokenizer,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir)
    save_temp_results(data)    


    
