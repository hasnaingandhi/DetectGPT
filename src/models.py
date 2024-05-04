import transformers
import torch
import time
from torch.nn import functional

class BaseModel:
    def __init__(self, base_model_name, device, cache_dir, logger):
        self.device = device
        self.cache_dir = cache_dir
        self.logger = logger
        logger.warning(f'Loading Base Model {base_model_name}')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def load_base_model(self, mask_model):
        self.logger.info("Moving Base Model to GPU...", end='', flush=True)
        start = time.time()
        try:
            mask_model.cpu()
        except NameError:
            pass
        self.model.to(self.device)
        self.logger.warning(f'Done ({time.time() - start:.2f}s)')

    
    
    def compute_ll(self, text):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
            labels = tokenized.input_ids
            return -self.model(**tokenized, labels=labels).loss.item()
        
    def compute_rank(self, text, log=False):
    # consider non-openai model

        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
            logits = self.model(**tokenized).logits[:, :-1]
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
    
    def compute_entropy(self, text):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
            logits = self.model(**tokenized).logits[:,:-1]
            neg_entropy = functional.softmax(logits, dim=-1) * functional.log_softmax(logits, dim=-1)
            return -neg_entropy.sum(-1).mean().item

 


class MaskModel:
    def __init__(self, mask_filling_model_name, device, model_max_length, cache_dir, logger):
        self.device = device
        self.cache_dir = cache_dir
        self.logger = logger
        logger.warning(f'Loading mask filling model {mask_filling_model_name}')
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name,
                                                                        torch_dtype=torch.bfloat16,
                                                                        load_in_8bit=True,
                                                                        device_map='auto',
                                                                        cache_dir=cache_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name,
                                                                    model_max_length=model_max_length,
                                                                    cache_dir=cache_dir
                                                                    )
    def load_mask_model(self, base_model):
        self.logger.warning('Moving Mask Model to GPU')
        start = time.time()

        base_model.cpu()
        self.model.to(self.device)
        self.logger.warning(f'Done ({time.time() - start:.2f}s)')