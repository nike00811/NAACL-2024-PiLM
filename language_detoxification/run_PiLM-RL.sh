python main.py --prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' --stepsize 0.3 --M 3 --N 100 --sample --ppl_weight '-0.10' --save_latent --seed 20 --output_dir "PiLM-RL-results/outputs_seed20"
python main.py --prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' --stepsize 0.3 --M 3 --N 100 --sample --ppl_weight '-0.10' --save_latent --seed 21 --output_dir "PiLM-RL-results/outputs_seed21"
python main.py --prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' --stepsize 0.3 --M 3 --N 100 --sample --ppl_weight '-0.10' --save_latent --seed 22 --output_dir "PiLM-RL-results/outputs_seed22"

python main.py --prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' --stepsize 0.3 --M 3 --N 100 --sample --ppl_weight '-0.10' --save_latent --seed 23 --output_dir "PiLM-RL-results-eval/outputs_seed23"

