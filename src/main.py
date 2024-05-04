import argparse
import os
import datetime
import json
import transformers
from transformers.utils import logging
import torch
import time
import datasets
from transformers import BitsAndBytesConfig
import random
import numpy as np
import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import functools
import utils
import loadingdata

API_TOKEN_COUNTER = 0
DEVICE = "cuda"
MODEL_MAX_LENGTH = 512
MIN_WORDS_SAMPLED = 55
TOP_K = 30
SAVE_FOLDER_NAME = ''
MIN_EXAMPLE_WORDS = 250


def save_temp_results(data):
    # Open file in write mode
    with open(os.path.join(SAVE_FOLDER_NAME, "temp_data.json"), "w") as file:
        # Write each item in the dataset to the file
        json.dump(data, file)

def load_mask_model():
    logger.warning('Moving mask model to GPU...')
    start = time.time()
    # for non-openai models
    base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f'Done ({time.time() - start:.2f}s)')
    
def create_save_folder(args, start_time):
    base_model_name = args.base_model.replace('/', '_')
    save_folder_name = f"results/{args.output_name}{base_model_name}-{args.mask_filling_model}/{start_time}-{args.pct_words_masked}-{args.n_perturbation_list}-{args.dataset}-{args.n_samples}"
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

def load_base_model(base_model):
    logger.info("Moving Base Model to GPU...", end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    base_model.to(DEVICE)
    logger.info(f'Done ({time.time() - start:.2f}s)')




def compute_ll(text):
    # consider non-openai model
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
    
def compute_rank(text, log=False):
    # consider non-openai model

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()
    
def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def baseline_experiment(criterion_func, name, n_samples=100):
    torch.manual_seed(1)
    np.random.seed(1)

    experiment_results = []
    for batch in tqdm.tqdm(range(n_samples // args.batch_size)):
        logger.warning(f"Computing {name} criterion")
        original_text = processed_data["original"][batch * args.batch_size:(batch + 1) * args.batch_size]
        sampled_text = processed_data["sampled"][batch * args.batch_size:(batch + 1) * args.batch_size]

        for idx in range(len(original_text)):
            experiment_results.append({
                "original": original_text[idx],
                "original_criterion": criterion_func(original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_criterion": criterion_func(sampled_text[idx])
            })
    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_criterion"] for x in experiment_results],
        'samples': [x["sampled_criterion"] for x in experiment_results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': experiment_results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }
    

def baseline_outputs():
    if not args.skip_baselines:
        outputs = [baseline_experiment(compute_ll, "log_likelihood", n_samples=args.n_samples)]
            
        get_rank = lambda text: -compute_rank(text, log=False)
        outputs.append(baseline_experiment(get_rank, "rank", n_samples=args.n_samples))
    return outputs

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens, end + args.buffer_size))
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def apply_perturbations(texts, span_length, pct, ceil_pct):
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), args.chunk_size)):
        logger.warning(f"Applying perturbations where number of perturbations = {n_perturbations}")
        outputs.extend(apply_perturbations(texts[i:i + args.chunk_size], span_length, pct, ceil_pct=ceil_pct))
        return outputs

def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=100):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = processed_data["original"]
    sampled_text = processed_data["sampled"]

    perturb_func = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)


if __name__ == '__main__':
    logging.set_verbosity_warning()
    logging.enable_explicit_format
    logger = logging.get_logger("transformers")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct-words-masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span-length', type=int, default=2)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--n-perturbation-list', type=str, default="1,10")
    parser.add_argument('--base-model', type=str, default="gpt2-medium")
    parser.add_argument('--mask-filling-model', type=str, default="t5-large")
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--chunk-size', type=int, default=20)
    parser.add_argument('--cache-dir', type=str, default=".cache")
    parser.add_argument('--output-name', type=str, default="n_perturb")
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--random-fills', action='store_true')
    parser.add_argument('--skip-baselines', action='store_true')
    args = parser.parse_args()

    logger.warning(args.dataset)
    logger.warning(args.baseline_only)
    logger.warning(args.random_fills)
    logger.warning(args.int8)

    torch.cuda.empty_cache()

    # considering fp16 precision 
    # considering top-k sampling for text generation

    formatted_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # save folder for results - 
    SAVE_FOLDER_NAME = create_save_folder(args, formatted_start_time)

    with open(os.path.join(SAVE_FOLDER_NAME, "arguments.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    setup_cache(args)
    perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    base_model, base_tokenizer = get_base_model(args.base_model, args.cache_dir)
    
    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)


    ######################################
    logger.warning("mask model")
    int8_kwargs = {}
    half_kwargs = dict(torch_dtype=torch.bfloat16)
    # Quantization configuration
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load the model with quantization
    # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)

    # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model)
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model, **int8_kwargs, **half_kwargs, cache_dir=args.cache_dir)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = MODEL_MAX_LENGTH

    logger.warning("done")
    #####################################
    
    logger.warning(f'Loading mask filling model {args.mask_filling_model}')
    
    # use sequence-to-sequence LM since we want to map the input text to a masked output text
    # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model, **kwargs, cache_dir=args.cache_dir)
    
    pre_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=MODEL_MAX_LENGTH, cache_dir=args.cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model, model_max_length=n_positions, cache_dir=args.cache_dir)

    load_base_model(base_model)
    torch.cuda.empty_cache()

    logger.warning(f'Loading dataset {args.dataset}...')
    object_processed_data = loadingdata.Dataset(args, logger, MIN_EXAMPLE_WORDS, MIN_WORDS_SAMPLED, MODEL_MAX_LENGTH, SAVE_FOLDER_NAME, DEVICE, pre_tokenizer, base_tokenizer, mask_model)
    processed_data = object_processed_data.dataset_generation()
    torch.cuda.empty_cache()
    logger.warning("Dataset loaded")

    base_outputs = baseline_outputs()
    save_temp_results(base_outputs)

    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations, n_samples)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(
                    perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
                outputs.append(output)
                with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
                    json.dump(output, f)
    torch.cuda.empty_cache()
    
    if not args.skip_baselines:
        # write likelihood threshold results to a file
        with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[0], f)

        if args.openai_model is None:
            # write rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[1], f)

            # write log rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[2], f)

            # write entropy threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[3], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)

        outputs += baseline_outputs
    torch.cuda.empty_cache()
    save_roc_curves(outputs)
    save_ll_histograms(outputs)
    save_llr_histograms(outputs)
    torch.cuda.empty_cache()

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)
    torch.cuda.empty_cache()
    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")
