
import numpy as np
import torch
import tqdm
import utils
import transformers
import re
import json
import os

class Experiments:
    def __init__(self, data, mask_model, base_model, batch_size, n_samples, n_perturbations, device, max_length, cache_dir, save_folder_name, span_length=2, pct_masked=0.3, zero_shot=True, supervised=True):
        self.zero_shot = zero_shot
        self.supervised = supervised
        self.baseline_results = []
        self.mask = mask_model
        self.base = base_model
        self.perturbation_results = []
        self.n_perturb = int(n_perturbations)
        self.data = data
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.span_length = span_length
        self.pct_masked = pct_masked
        self.device = device
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.save_folder = save_folder_name

    def baseline_threshold_results(self, criterion_func, name):
        torch.manual_seed(0)
        np.random.seed(0)

        results = []
        for batch in tqdm.tqdm(range(self.n_samples // self.batch_size), desc=f"Computing {name} criterion"):
            original_text = self.data["original"][batch * self.batch_size:(batch + 1) * self.batch_size]
            sampled_text = self.data["sampled"][batch * self.batch_size:(batch + 1) * self.batch_size]

            for idx in range(len(original_text)):
                results.append({
                    "original": original_text[idx],
                    "original_criterion": criterion_func(original_text[idx]),
                    "sampled": sampled_text[idx],
                    "sampled_criterion": criterion_func(sampled_text[idx]),
                })

        predictions = {
            'real': [x["original_criterion"] for x in results],
            'samples': [x["sampled_criterion"] for x in results],
        }

        fpr, tpr, roc_auc = utils.get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = utils.get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
        return {
            'name': f'{name}_threshold',
            'predictions': predictions,
            'info': {
                'n_samples': self.n_samples,
            },
            'raw_results': results,
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
    
    def run_supervised(self, model):
        print(f'Beginning supervised evaluation with {model}...')
        detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=self.cache_dir).to(self.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=self.cache_dir)

        real, fake = self.data['original'], self.data['sampled']

        with torch.no_grad():
            real_preds = []
            for batch in tqdm.tqdm(range(len(real) // self.batch_size), desc="Evaluating real"):
                batch_real = real[batch * self.batch_size:(batch + 1) * self.batch_size]
                batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
            
            fake_preds = []
            for batch in tqdm.tqdm(range(len(fake) // self.batch_size), desc="Evaluating fake"):
                batch_fake = fake[batch * self.batch_size:(batch + 1) * self.batch_size]
                batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

        predictions = {
            'real': real_preds,
            'samples': fake_preds,
        }

        fpr, tpr, roc_auc = utils.get_roc_metrics(real_preds, fake_preds)
        p, r, pr_auc = utils.get_precision_recall_metrics(real_preds, fake_preds)
        print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

        del detector
        torch.cuda.empty_cache()

        return {
            'name': model,
            'predictions': predictions,
            'info': {
                'n_samples': self.n_samples,
            },
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

    def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = pct * len(tokens) / (span_length + 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - 1)
            search_end = min(len(tokens), end + 1)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text


    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask.tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.mask.model.generate(**tokens, max_length=150, do_sample=True, top_p=1, num_return_sequences=1, eos_token_id=stop_id)
        return self.mask.tokenizer.batch_decode(outputs, skip_special_tokens=False)


    def extract_fills(self, texts):
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        extracted_fills = [re.compile(r"<extra_id_\d+>").split(x)[1:-1] for x in texts]
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills


    def apply_extracted_fills(self, masked_texts, extracted_fills):
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        texts = [" ".join(x) for x in tokens]
        return texts


    def tokenize_and_mask(self, text, ceil_pct=False):
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = self.pct_masked * len(tokens) / (self.span_length + 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - self.span_length)
            end = start + self.span_length
            search_start = max(0, start - 1)
            search_end = min(len(tokens), end + 1)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def perturb_texts(self, texts, ceil_pct=False):
        masked_texts = [self.tokenize_and_mask(x) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        return perturbed_texts


    def start_perturb(self, text):
        chunk_size = 20
        perturbed_output = []
        for i in tqdm.tqdm(range(0, len(text), chunk_size), desc="Applying perturbations"):
            perturbed_output.extend(self.perturb_texts(text[i:i + chunk_size]))
        return perturbed_output

    def apply_perturbations(self):
        self.mask.load_mask_model(self.base)

        torch.manual_seed(0)
        np.random.seed(0)

        results = []
        original = self.data["original"]
        sampled = self.data["sampled"]

        perturb_sampled = self.start_perturb([x for x in sampled for _ in range(self.n_perturb)])
        perturb_original = self.start_perturb([x for x in original for _ in range(self.n_perturb)])

        assert len(perturb_sampled) == len(sampled) * self.n_perturb, f"Expected {len(sampled) * self.n_perturb} perturbed samples, got {len(perturb_sampled)}"
        assert len(perturb_original) == len(original) * self.n_perturb, f"Expected {len(original) * self.n_perturb} perturbed samples, got {len(perturb_original)}"

        for idx in range(len(original)):
            results.append({
                "original": original[idx],
                "sampled": sampled[idx],
                "perturbed_sampled": perturb_sampled[idx * self.n_perturb: (idx + 1) * self.n_perturb],
                "perturbed_original": perturb_original[idx * self.n_perturb: (idx + 1) * self.n_perturb]
            })

        self.base.load_base_model(self.mask)

        for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
            perturb_sampled_ll = [self.base.compute_ll(sample) for sample in (res["perturbed_sampled"])]
            perturb_original_ll = [self.base.compute_ll(sample) for sample in (res["perturbed_original"])]
            res["original_ll"] = self.base.compute_ll(res["original"])
            res["sampled_ll"] = self.base.compute_ll(res["sampled"])
            res["all_perturbed_sampled_ll"] = perturb_sampled_ll
            res["all_perturbed_original_ll"] = perturb_original_ll
            res["perturbed_sampled_ll"] = np.mean(perturb_sampled_ll)
            res["perturbed_original_ll"] = np.mean(perturb_original_ll)
            res["perturbed_sampled_ll_std"] = np.std(perturb_sampled_ll) if len(perturb_sampled_ll) > 1 else 1
            res["perturbed_original_ll_std"] = np.std(perturb_original_ll) if len(perturb_original_ll) > 1 else 1
        
        return results
    
    def run_perturbation_experiment(self, results, criterion):
        predictions = {'real': [], 'samples': []}
        for res in results:
            if criterion == 'd':
                predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
                predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
            elif criterion == 'z':
                if res['perturbed_original_ll_std'] == 0:
                    res['perturbed_original_ll_std'] = 1
                    print("WARNING: std of perturbed original is 0, setting to 1")
                    print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                    print(f"Original text: {res['original']}")
                if res['perturbed_sampled_ll_std'] == 0:
                    res['perturbed_sampled_ll_std'] = 1
                    print("WARNING: std of perturbed sampled is 0, setting to 1")
                    print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                    print(f"Sampled text: {res['sampled']}")
                predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
                predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

        fpr, tpr, roc_auc = utils.get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = utils.get_precision_recall_metrics(predictions['real'], predictions['samples'])
        name = f'perturbation_{self.n_perturb}_{criterion}'
        print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
        return {
            'name': name,
            'predictions': predictions,
            'info': {
                'pct_words_masked': self.pct_masked,
                'span_length': self.span_length,
                'n_perturbations': self.n_perturb,
                'n_samples': self.n_samples,
            },
            'raw_results': results,
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





    def run(self, base_model, baseline=True, perturbation=True):
        self.baseline_results = [self.baseline_threshold_results(base_model.compute_ll, "likelihood")]

        if self.zero_shot and baseline:
            rank_criterion = lambda text: -base_model.compute_rank(text, log=False)
            self.baseline_results.append(self.baseline_threshold_results(rank_criterion, "rank"))
            logrank_criterion = lambda text: -base_model.compute_rank(text, log=True)
            self.baseline_results.append(self.baseline_threshold_results(logrank_criterion, "log_rank"))
            entropy_criterion = lambda text: base_model.compute_entropy(text)
            self.baseline_results.append(self.baseline_threshold_results(entropy_criterion, "entropy"))

        if self.supervised and baseline:
            self.baseline_results.append(self.run_supervised(model='roberta-base-openai-detector'))
            self.baseline_results.append(self.run_supervised(model='roberta-large-openai-detector'))

        if perturbation:
            perturbed_data = self.apply_perturbations()
            for perturbation_mode in ['d', 'z']:
                result = self.run_perturbation_experiment(
                    perturbed_data, perturbation_mode
                )
                self.perturbation_results.append(result)
                with open(os.path.join(self.save_folder, f"perturbation_{self.n_perturb}_{perturbation_mode}_results.json"), "w") as f:
                        json.dump(result, f)
        

