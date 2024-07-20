import sys
sys.path.append('../src')
from method import set_seed, Plug_in_LM
from metrics import get_PPL

import argparse
import torch
import os
import json
import pickle
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_controller import Regression_Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model",       type=str,   default="gpt2-medium", help="pretrained model name or path to local checkpoint", )
    parser.add_argument("--pretrained_tokenizer",   type=str,   default="gpt2-medium", help="pretrained tokenizer name or path to local checkpoint", )
    parser.add_argument('--internal_classifier',   type=str,   default='unitary/toxic-bert')
    parser.add_argument("--device", type=str, default='cuda')
    


    parser.add_argument('--prefix_set', type=str, default='../data/prefixes/top100-continuation-real_toxicity_prompts.json')
    parser.add_argument("--length",                 type=int,   default=50)
    parser.add_argument("--stepsize",               type=float, default=0.09)
    parser.add_argument('--M',                      type=int,   default=3, help='number of update hidden state')
    parser.add_argument('--N',                      type=int,   default=3, help='number of sampling in RL')
    parser.add_argument("--future_n_tokens",   type=int,   default=10)
    parser.add_argument("--generate_n_tokens",   type=int,   default=10)
    parser.add_argument('--ppl_weight',  type=float, default=0)
    
    parser.add_argument("--sample",  action="store_true")
    parser.add_argument("--gamma",                  type=float, default=1.0)
    parser.add_argument("--gm_scale",               type=float, default=1.0)
    parser.add_argument("--bias_method", type=str, default='mean', choices=['mean', 'none'])
    parser.add_argument("--bias_location", type=str, default='previous', choices=['current', 'previous'])
    parser.add_argument("--modify_width",  type=int, default=100, help='freeze width')
    parser.add_argument("--topk",  type=int, default=10, help='topk sampling')
    parser.add_argument('--use_prefix_reward', action='store_true')
    parser.add_argument('--reset_prefix_hidden', action='store_true')
    parser.add_argument('--dynamic_M', action='store_true')
    parser.add_argument('--debug',        action="store_true")
    parser.add_argument('--replacement',  action="store_true")
    parser.add_argument('--use_prompt',   type=int, default=0)
    parser.add_argument('--keep_prompt',     action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    

    parser.add_argument('--save_latent',     action='store_true')
    parser.add_argument("--seed",                   type=int,   default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    
    parser.add_argument('--use_controller', action='store_true')
    parser.add_argument('--controller', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.controller is not None:
        args.use_controller = True
    elif args.use_controller:
        assert args.controller is not None

    if args.keep_prompt:
        assert args.use_prompt > 0
        assert args.reset_prefix_hidden

    if args.save_latent:
        assert args.output_dir is not None
        
    if not torch.cuda.is_available():
        args.device = 'cpu'
    
    return args




def load_models(args):
    # set Random seed
    set_seed(args.seed)

    # load pretrained model

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        output_hidden_states=True
    )
    model.to(args.device)
    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)
    
    internal_classifier = AutoModelForSequenceClassification.from_pretrained(args.internal_classifier).to(args.device)
    internal_classifier.eval()
    internal_classifier_tokenizer = AutoTokenizer.from_pretrained(args.internal_classifier)
    
    
    if args.use_controller:
        latent_controller = torch.load(args.controller).to(args.device)
        latent_controller.eval()
    else:
        latent_controller = None

    models = {}
    models['model'] = model
    models['tokenizer'] = tokenizer
    
    models['internal_classifier'] = internal_classifier
    models['internal_classifier_tokenizer'] = internal_classifier_tokenizer
    
    models['latent_controller'] = latent_controller
    
    return models




@torch.no_grad()
def get_toxic_probability(text, clsf_model, clsf_token):
    inputs = clsf_token(text, return_tensors='pt').to(clsf_model.device)
    outputs = clsf_model(**inputs)
    output_score = outputs.logits[0].sigmoid()
    toxic_probability = output_score[ clsf_model.config.label2id['toxic'] ]
    return toxic_probability.item()



class PiLM_detoxic(Plug_in_LM):
    def __init__(self, args, model, tokenizer, internal_classifier, internal_classifier_tokenizer, latent_controller):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.internal_classifier = internal_classifier
        self.internal_classifier_tokenizer = internal_classifier_tokenizer
        self.latent_controller = latent_controller

    @torch.no_grad()
    def get_reward(self, batch_text, prefix_text=''):
        if type(batch_text) == str:
            batch_text = [batch_text]
        batch_scores = []
        for text in batch_text:
            inputs = self.internal_classifier_tokenizer(prefix_text+text, return_tensors='pt').to(args.device)
            outputs = self.internal_classifier(**inputs)
            output_score = outputs.logits[0].sigmoid()
#             {0: 'toxic',
#              1: 'severe_toxic',
#              2: 'obscene',
#              3: 'threat',
#              4: 'insult',
#              5: 'identity_hate'}
#             score = output_score[0]
            score = 1 - output_score[ self.internal_classifier.config.label2id['toxic'] ]
            
            if self.args.ppl_weight != 0:
                ppl = get_PPL(prefix_text+text, self.model, self.tokenizer)
                ppl = torch.tensor(ppl)
                score = score + self.args.ppl_weight * ppl.log()
            
            batch_scores.append(score)
        batch_scores = torch.stack(batch_scores)
        return batch_scores

    def check_good_enough(self, text):
#         if self.args.early_stop:
#             inputs = self.internal_classifier_tokenizer(text, return_tensors='pt').to(args.device)
#             output_score = self.internal_classifier(**inputs).logits[0].softmax(dim=-1)
#             # output_score.shape = [negative, neutral, positive]
#             score = output_score[ internal_classifier.config.label2id[self.args.sentiment] ]
#             if score > 0.9:
#                 return True
        return False




if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    with open(args.prefix_set) as reader:
        prefixes_set = json.load(reader)['prefix']

    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        candidate = ['{:03d}.json'.format(i) for i in range(len(prefixes_set))]
        candidate = set(candidate)
        if candidate.issubset(os.listdir(args.output_dir)):
            print('[INFO] {} already done'.format(args.output_dir))
            exit()

    if 'models' not in locals():
        print('loading models')
        models = load_models(args)
        model                = models['model']
        tokenizer            = models['tokenizer']
        internal_classifier = models['internal_classifier']
        internal_classifier_tokenizer  = models['internal_classifier_tokenizer']
        
        latent_controller = models['latent_controller']
    

    PiLM = PiLM_detoxic(
                      args=args,
                      model=model,
                      tokenizer=tokenizer,
                      internal_classifier=internal_classifier,
                      internal_classifier_tokenizer=internal_classifier_tokenizer,
                      latent_controller=latent_controller,
                     )


    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    for index, prefix_text in enumerate(prefixes_set):
        if args.output_dir is not None:
            setting = '{:03d}.json'.format(index)
            output_path = '{}/{}'.format(args.output_dir, setting)

            if os.path.exists(output_path):
                print('[INFO] SKIP')
                continue

            with open(output_path, mode='w', encoding='utf-8') as writer:
                print('Not yet', file=writer)

        print('Prefix: {}'.format(prefix_text))

        set_seed(args.seed)
        PiLM_outputs = PiLM.generate_text(prefix_text)

        output_text = tokenizer.decode(PiLM_outputs['output_ids'][0])
        print(output_text)

        output_text = PiLM.extract_text(output_text)

        toxic_probabiity = get_toxic_probability(output_text, internal_classifier, internal_classifier_tokenizer)
        print('toxic_probabiity: {}'.format(toxic_probabiity))

        obj = {
               'args': vars(args),
               'output_text': output_text,
               'output_ids': PiLM_outputs['output_ids'][0].tolist(),

               'internal_toxic': toxic_probabiity,

               'step_record': PiLM_outputs['step_record'].tolist(),
               'time_record': PiLM_outputs['time_record'].tolist(),
              }

        if args.output_dir is not None:
            with open(output_path, mode='w', encoding='utf-8') as fp:
                json.dump(obj=obj, fp=fp, indent=4)

            if args.save_latent:
                with open(output_path.replace('.json', '.pickle'), mode='wb') as fp:
                    pickle.dump(PiLM_outputs['latent_collect'], file=fp)

