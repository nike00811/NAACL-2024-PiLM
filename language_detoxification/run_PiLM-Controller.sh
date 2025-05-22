python main.py \
--prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' \
--sample  \
--use_controller \
--controller "Controllers/controller-detoxic/final_model.pt" \
--seed 20 \
--output_dir "PiLM-Controller-results/outputs_seed20"

python main.py \
--prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' \
--sample  \
--use_controller \
--controller "Controllers/controller-detoxic/final_model.pt" \
--seed 21 \
--output_dir "PiLM-Controller-results/outputs_seed21"

python main.py \
--prefix_set '../data/prefixes/top100-continuation-real_toxicity_prompts.json' \
--sample  \
--use_controller \
--controller "Controllers/controller-detoxic/final_model.pt" \
--seed 22 \
--output_dir "PiLM-Controller-results/outputs_seed22"
