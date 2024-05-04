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
from models import BaseModel, MaskModel
from transformers import BitsAndBytesConfig
import experiments as exp

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
    
def create_save_folder(args, start_time):
    base_model_name = args.base_model.replace('/', '_')
    save_folder_name = f"results/{args.output_name}{base_model_name}-{args.mask_filling_model}/{start_time}-{args.pct_words_masked}-{args.n_perturbations}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    logger.info(f"Saving results to path: {os.path.abspath(save_folder_name)}")
    return save_folder_name

def setup_cache(args):
    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    logger.info(f"Cache directory: {args.cache_dir}")


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
    parser.add_argument('--n-perturbations', type=str, default="100")
    parser.add_argument('--base-model', type=str, default="gpt2-medium")
    parser.add_argument('--mask-filling-model', type=str, default="t5-large")
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--chunk-size', type=int, default=20)
    parser.add_argument('--cache-dir', type=str, default=".cache")
    parser.add_argument('--output-name', type=str, default="n_perturb")
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--random-fills', action='store_true')
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='.cache')
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # considering fp16 precision 
    # considering top-k sampling for text generation

    formatted_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # save folder for results - 
    SAVE_FOLDER_NAME = create_save_folder(args, formatted_start_time)
    base_model_name = args.base_model.replace('/', '_')
    mask_model_name = args.mask_filling_model.replace('/', '_')

    with open(os.path.join(SAVE_FOLDER_NAME, "arguments.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    setup_cache(args)
    base_model = BaseModel(args.base_model, DEVICE, args.cache_dir, logger)
    
    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
    mask_model = MaskModel(args.mask_filling_model, DEVICE, MODEL_MAX_LENGTH, args.cache_dir, logger)

    
    pre_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=MODEL_MAX_LENGTH, cache_dir=args.cache_dir)

    base_model.load_base_model(mask_model)
    torch.cuda.empty_cache()

    logger.warning(f'Loading dataset {args.dataset}...')
    processed_data = loadingdata.Dataset(args, logger, SAVE_FOLDER_NAME, args.batch_size)
    processed_data = processed_data.dataset_generation(pre_tokenizer, base_model, MIN_EXAMPLE_WORDS, MODEL_MAX_LENGTH, MIN_WORDS_SAMPLED)
    torch.cuda.empty_cache()
    logger.warning("Dataset loaded")

    logger.warning("Starting Baseline Experiments...")
    experiments = exp.Experiments(processed_data,
                                           mask_model,
                                           base_model,
                                           args.batch_size,
                                           args.n_samples,
                                           args.n_perturbations, 
                                           DEVICE,
                                           MODEL_MAX_LENGTH,
                                           args.cache_dir,
                                           SAVE_FOLDER_NAME
                                           )
    experiments.run(base_model)
    save_temp_results(experiments.baseline_results)
    logger.warning("Finished baseline experiments")

    utils.save_graphs(experiments.baseline_results, base_model_name, mask_model_name, SAVE_FOLDER_NAME)

    