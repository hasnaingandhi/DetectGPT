import transformers
import torch

class BaseModel:
    def __init__(self, base_model_name, device, cache_dir, logger):
        self.device = device
        self.cache_dir = cache_dir
        self.logger = logger
        logger.warning(f'Loading Base Model {base_model_name}')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


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