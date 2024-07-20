python train_controller.py \
--train_dir  'PiLM-RL-results/outputs_seed*'  \
--eval_dir   'PiLM-RL-results-eval/outputs_seed*' \
--attribute  'positive' \
--output_dir "Controllers/controller-positive"

python train_controller.py \
--train_dir  'PiLM-RL-results/outputs_seed*'  \
--eval_dir   'PiLM-RL-results-eval/outputs_seed*' \
--attribute  'negative' \
--output_dir "Controllers/controller-negative"
