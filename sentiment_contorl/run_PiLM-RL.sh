python main.py \
--prefix_set "../data/prefixes/sentiment_prefixes.json" \
--stepsize 0.45 \
--M 2 \
--N 100 \
--length 50 \
--future_n_tokens 10 \
--generate_n_tokens 10 \
--sample \
--seed 20 \
--save_latent \
--ppl_weight '-0.05' \
--output_dir "PiLM-RL-results/outputs_seed20"

python main.py \
--prefix_set "../data/prefixes/sentiment_prefixes.json" \
--stepsize 0.45 \
--M 2 \
--N 100 \
--length 50 \
--future_n_tokens 10 \
--generate_n_tokens 10 \
--sample \
--seed 21 \
--save_latent \
--ppl_weight '-0.05' \
--output_dir "PiLM-RL-results/outputs_seed21"

python main.py \
--prefix_set "../data/prefixes/sentiment_prefixes.json" \
--stepsize 0.45 \
--M 2 \
--N 100 \
--length 50 \
--future_n_tokens 10 \
--generate_n_tokens 10 \
--sample \
--seed 22 \
--save_latent \
--ppl_weight '-0.05' \
--output_dir "PiLM-RL-results/outputs_seed22"


python main.py \
--prefix_set "../data/prefixes/sentiment_prefixes.json" \
--stepsize 0.45 \
--M 2 \
--N 100 \
--length 50 \
--future_n_tokens 10 \
--generate_n_tokens 10 \
--sample \
--seed 23 \
--save_latent \
--ppl_weight '-0.05' \
--output_dir "PiLM-RL-results-eval/outputs_seed23"
