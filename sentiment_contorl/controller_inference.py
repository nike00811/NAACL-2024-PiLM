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

from main import parse_args, load_models, get_sentiment_score



class PiLM_setniment(Plug_in_LM):
    def __init__(self, args, model, tokenizer, internal_classifier, internal_classifier_tokenizer, latent_controller):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.internal_classifier = internal_classifier
        self.internal_classifier_tokenizer = internal_classifier_tokenizer
        self.latent_controller = latent_controller

    @torch.no_grad()
    def get_reward(self, batch_text, prefix_text=''):
        return torch.tensor([0])

    def check_good_enough(self, text):
        if self.args.early_stop:
            inputs = self.internal_classifier_tokenizer(text, return_tensors='pt').to(args.device)
            output_score = self.internal_classifier(**inputs).logits[0].softmax(dim=-1)
            # output_score.shape = [negative, neutral, positive]
            score = output_score[ internal_classifier.config.label2id[self.args.sentiment] ]
            if score > 0.9:
                return True
        return False


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    with open(args.prefix_set) as reader:
        prefixes_set = json.load(reader)['prefix']

#     sentiment_attribute = ['positive', 'negative']
    
    if args.use_controller:
        controller_attribute = args.controller.split('/')[-2].split('-')[-1]
        sentiment_attribute = [controller_attribute]
    

    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
        candidate = []
        for sentiment in sentiment_attribute:
            for prefix_text in prefixes_set:
                setting = '{}+{}.json'.format(sentiment, prefix_text.replace(' ', '_'))
                candidate.append(setting)
        candidate = set(candidate)
        
        if candidate.issubset(os.listdir(args.output_dir)):
            print('[INFO] {} already done'.format(args.output_dir))
            exit()


    print('loading models')
    models = load_models(args)
    model                = models['model']
    tokenizer            = models['tokenizer']
    internal_classifier = models['internal_classifier']
    internal_classifier_tokenizer  = models['internal_classifier_tokenizer']
    
    external_classifier  = models['external_classifier']
    external_classifier_tokenizer  = models['external_classifier_tokenizer']
    latent_controller = models['latent_controller']
        
    PiLM = PiLM_setniment(
                      args=args,
                      model=model,
                      tokenizer=tokenizer,
                      internal_classifier=internal_classifier,
                      internal_classifier_tokenizer=internal_classifier_tokenizer,
                      latent_controller=latent_controller,
                     )


    for sentiment in sentiment_attribute:
        for prefix_text in prefixes_set:
            if args.output_dir is not None:
                setting = '{}+{}.json'.format(sentiment, prefix_text.replace(' ', '_'))
                output_path = '{}/{}'.format(args.output_dir, setting)

                if os.path.exists(output_path):
                    print('[INFO] SKIP')
                    continue

                with open(output_path, mode='w', encoding='utf-8') as writer:
                    print('Not yet', file=writer)
            
            args.sentiment = sentiment

            print('sentiment: {}'.format(args.sentiment))
            print('Prefix: {}'.format(prefix_text))
            
            set_seed(args.seed)
            PiLM_outputs = PiLM.generate_text(prefix_text)

            output_text = tokenizer.decode(PiLM_outputs['output_ids'][0])
            print(output_text)

            output_text = PiLM.extract_text(output_text)
            
            intcls_score = get_sentiment_score(output_text, internal_classifier, internal_classifier_tokenizer)
            extcls_score = get_sentiment_score(output_text, external_classifier, external_classifier_tokenizer)
            print('intcls_score: {}'.format(intcls_score))
            print('extcls_score: {}'.format(extcls_score))
            
            obj = {
                   'args': vars(args),
                   'output_text': output_text,
                   'output_ids': PiLM_outputs['output_ids'][0].tolist(),
                   
                   'internal_score': intcls_score,
                   'external_score': extcls_score,
                   
                   'step_record': PiLM_outputs['step_record'].tolist(),
                   'time_record': PiLM_outputs['time_record'].tolist(),
                  }

            if args.output_dir is not None:
                with open(output_path, mode='w', encoding='utf-8') as fp:
                    json.dump(obj=obj, fp=fp, indent=4)
                
                if args.save_latent:
                    with open(output_path.replace('.json', '.pickle'), mode='wb') as fp:
                        pickle.dump(PiLM_outputs['latent_collect'], file=fp)

