#!/bin/bash -l
#SBATCH --output=outputfile

python main.py --output-name n_perturb --base-model gpt2-xl --mask-filling-model t5-small --n-perturbation-list 1,10 --n-samples 100 --pct-words-masked 0.3 --span-length 2 --cache-dir ".cache"