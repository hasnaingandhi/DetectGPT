import datasets
import random
import torch
import numpy as np
import json
import os
import utils



class Dataset:
    def __init__(self, args, logger, save_folder_name, batch_size):
        self.logger = logger
        self.args = args
        self.save_folder_name = save_folder_name
        self.batch_size = batch_size

    def get_data(self, pre_tokenizer, base_model, min_example_words, model_max_length, min_words_sampled):
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
            long_samples = [x for x in data if len(x.split()) > min_example_words]
            if len(long_samples) > 0:
                data = long_samples

        random.seed(1)
        random.shuffle(data)
        data = data[:5000]

        tokenized_data = pre_tokenizer(data)
        data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= model_max_length]

        self.logger.warning(f"Total number of samples: {len(data)}")
        self.logger.warning(f"Avg number of words: {np.mean([len(x.split()) for x in data])}")

        return self.get_samples(data[:self.args.n_samples], base_model, min_words_sampled)

    
    def get_samples(self, raw, base_model, min_words_sampled):
        # torch.cuda.empty_cache()
        self.logger.info(f"Generating samples...")
        torch.manual_seed(30)
        np.random.seed(30)
        data = {
            "original": [],
            "sampled": [],
        }
        for batch in range(len(raw) // self.batch_size):
            self.logger.info(f"Generating samples for batch {batch} of {len(raw) // self.batch_size}")
            original = raw[batch * self.batch_size: (batch+1)*self.batch_size]
            sampled = base_model.sample_from_model(original, min_words=min_words_sampled)

            for o, s in zip(original, sampled):
                o, s = utils.trim_to_shortest(o, s)
                data["original"].append(o)
                data["sampled"].append(s)

        return data

    def dataset_generation(self, pre_tokenizer, base_model, min_example_words, model_max_length, min_words_sampled):
        data = self.get_data(pre_tokenizer, base_model, min_example_words, model_max_length, min_words_sampled)
        if self.args.random_fills:
            FILL_DICTIONARY = set()
            for texts in data.values():
                for text in texts:
                    FILL_DICTIONARY.update(text.split())
            FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

        with open(os.path.join(self.save_folder_name, "raw_data.json"), "w") as f:
            self.logger.warning(f"Writing raw data to {os.path.join(self.save_folder_name, 'raw_data.json')}")
            json.dump(data, f)
        
        return data

