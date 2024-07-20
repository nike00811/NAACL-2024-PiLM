python controller_inference.py --length 50 --sample --controller "Controllers/controller-positive/final_model.pt" --seed 20 --output_dir "PiLM-Controller-results/outputs_seed20"
python controller_inference.py --length 50 --sample --controller "Controllers/controller-negative/final_model.pt" --seed 20 --output_dir "PiLM-Controller-results/outputs_seed20"

python controller_inference.py --length 50 --sample --controller "Controllers/controller-positive/final_model.pt" --seed 21 --output_dir "PiLM-Controller-results/outputs_seed21"
python controller_inference.py --length 50 --sample --controller "Controllers/controller-negative/final_model.pt" --seed 21 --output_dir "PiLM-Controller-results/outputs_seed21"

python controller_inference.py --length 50 --sample --controller "Controllers/controller-positive/final_model.pt" --seed 22 --output_dir "PiLM-Controller-results/outputs_seed22"
python controller_inference.py --length 50 --sample --controller "Controllers/controller-negative/final_model.pt" --seed 22 --output_dir "PiLM-Controller-results/outputs_seed22"
