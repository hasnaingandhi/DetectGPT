import datasets
import random
import torch
import numpy as np
import tqdm
import functools
import transformers
from transformers.utils import logging


class Dataset:
    def __init__(self, args, logger, MIN_EXAMPLE_WORDS, MODEL_MAX_LENGTH, SAVE_FOLDER_NAME):
        self.logger = logger
        self.args = args
        self.MIN_EXAMPLE_WORDS = MIN_EXAMPLE_WORDS
        self.MODEL_MAX_LENGTH = MODEL_MAX_LENGTH
        self.SAVE_FOLDER_NAME = SAVE_FOLDER_NAME

    def get_data(self, pre_tokenizer):
        data = datasets.load_dataset(self.args.dataset, self.args.cache_dir, split="train")[self.args.dataset_key]

        # PREPROCESS
        # remove duplicates
        data = list(dict.fromkeys(data))

        # remove new lines
        data = [' '.join(x.split()) for x in data]

        # remove surrounding whitespace
        data = [x.strip() for x in data]

        # Keep sufficiently long examples
        if self.args.dataset=='xsum' :
            long_samples = [x for x in data if len(x.split()) > self.MIN_EXAMPLE_WORDS]
            if len(long_samples) > 0:
                data = long_samples

        random.seed(1)
        random.shuffle(data)
        data = data[:1000]

        tokenized_data = pre_tokenizer(data)
        data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= self.MODEL_MAX_LENGTH]

        self.logger.warning(f"Total number of samples: {len(data)}")
        self.logger.warning(f"Avg number of words: {np.mean([len(x.split()) for x in data])}")

        return self.get_samples(self, data[:self.args.n_samples])

    def get_samples(self, raw):
        # torch.cuda.empty_cache()
        # GENERATE SAMPLES
        self.logger.info(f"Generating samples...")
        torch.manual_seed(31)
        np.random.seed(31)
        data = {
            "original": [],
            "sampled": [],
        }
        for batch in range(len(raw) // args.batch_size):
            self.logger.info(f"Generating samples for batch {batch} of {len(raw) // args.batch_size}")
            original = raw[batch*args.batch_size: (batch+1)*args.batch_size]
            sampled = sample_from_model(original, min_words=self.args.MIN_WORDS_SAMPLED)

            for o, s in zip(original, sampled):
                o, s = utils.trim_to_shortest(o, s)
                data["original"].append(o)
                data["sampled"].append(s)

        return data

    def dataset_generation(self, pre_tokenizer):
        data = self.get_data(pre_tokenizer)
        if self.args.random_fills:
            FILL_DICTIONARY = set()
            for texts in data.values():
                for text in texts:
                    FILL_DICTIONARY.update(text.split())
            FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

        with open(os.path.join(self.SAVE_FOLDER_NAME, "raw_data.json"), "w") as f:
            self.logger.warning(f"Writing raw data to {os.path.join(self.SAVE_FOLDER_NAME, 'raw_data.json')}")
            json.dump(data, f)
        
        return data

