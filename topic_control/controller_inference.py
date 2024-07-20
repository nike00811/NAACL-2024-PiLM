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

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import spacy
import colorama


from train_controller import Regression_Model
from main import (
    parse_args,
    add_color,
    colorful,
    load_models,
    load_wordlists,
)


class PiLM_topic(Plug_in_LM):
    def __init__(self, args, model, tokenizer, spacy_model, topic_embedding, latent_controller):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.spacy_model = spacy_model
        self.topic_embedding = topic_embedding
        
        self.latent_controller = latent_controller
        
        self.get_cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.reward_topk = 2

    def lemmatization(self, text):
        tokens = self.spacy_model(text)
        lemma = ''.join([token.lemma_+token.whitespace_ for token in tokens])
        return lemma


    def get_embedding(self, text):
        lemma_text = self.lemmatization(text.lower()) # Use lowercase to calculate rewards
        tokens = self.spacy_model(lemma_text)
        embedding = []
        for token in tokens:
            embedding.append(token.vector)
        # type(embedding) = list of np.array
        embedding = np.array(embedding)
        # type(embedding) = 2D np.array
        embedding = torch.tensor(embedding)
        # type(embedding) = 2D torch.tensor
        return embedding

    @torch.no_grad()
    def get_reward(self, batch_text, prefix_text=''):
        if type(batch_text) == str:
            batch_text = [batch_text]
        
        topic_embed = self.topic_embedding[self.args.topic]
        prefix_reward = []
        if prefix_text != '':
            prefix_reward = []
            prefix_embed = self.get_embedding(prefix_text)
            for i in prefix_embed:
                match = self.get_cosine(i, topic_embed).max()
                prefix_reward.append(match)

            prefix_reward = torch.stack(prefix_reward).topk(min(self.reward_topk, len(prefix_reward))).values
        
        
        batch_scores = []
        for text in batch_text:
            embed = self.get_embedding(text)
            
            score = []
            score += list(prefix_reward)
            
            for i in embed:
                match = self.get_cosine(i, topic_embed).max()
                score.append(match)
            score = torch.stack(score).topk(min(self.reward_topk, len(score))).values.sum()
            
            if self.args.ppl_weight != 0:
                ppl = get_PPL(prefix_text+text, self.model, self.tokenizer)
                ppl = torch.tensor(ppl)
                score = score + self.args.ppl_weight * ppl.log()
            
            batch_scores.append(score)
            
        batch_scores = torch.stack(batch_scores)
        return batch_scores

    def check_good_enough(self, text):
        topic_embed = self.topic_embedding[self.args.topic]
        score = []
        embed = self.get_embedding(text)
        for i in embed:
            match = self.get_cosine(i, topic_embed).max()
            score.append(match)
        score = torch.stack(score).topk(min(self.reward_topk, len(score))).values.sum()
        if score > 1.99:
            return True
        return False



if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    with open(args.prefix_set) as reader:
        prefixes_set = json.load(reader)['prefix']

    if args.use_controller:
        controller_topic = args.controller.split('/')[-2].split('-')[-1]
        topic_attribute = [controller_topic]
    

    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
        candidate = []
        for topic in topic_attribute:
            for prefix_text in prefixes_set:
                setting = '{}+{}.json'.format(topic, prefix_text.replace(' ', '_'))
                candidate.append(setting)
        candidate = set(candidate)
        
        if candidate.issubset(os.listdir(args.output_dir)):
            print('[INFO] {} already done'.format(args.output_dir))
            exit()

    if 'models' not in locals():
        print('loading models')
        models = load_models(args)
        model            = models['model']
        tokenizer        = models['tokenizer']
        spacy_model = models['spacy_model']
        latent_controller = models['latent_controller']

    
    
    wordlists = load_wordlists(args.data_dir)
    PiLM = PiLM_topic(
                      args=args,
                      model=model,
                      tokenizer=tokenizer,
                      spacy_model=spacy_model,
                      topic_embedding=wordlists['topic_embedding'],
                      latent_controller=latent_controller,
                     )


    for topic in topic_attribute:
        for prefix_text in prefixes_set:
            if args.output_dir is not None:
                setting = '{}+{}.json'.format(topic, prefix_text.replace(' ', '_'))
                output_path = '{}/{}'.format(args.output_dir, setting)

                if os.path.exists(output_path):
                    print('[INFO] SKIP')
                    continue

                with open(output_path, mode='w', encoding='utf-8') as writer:
                    print('Not yet', file=writer)
            
            args.topic = topic

            print('topic: {}'.format(args.topic))
            print('Prefix: {}'.format(prefix_text))

            
            set_seed(args.seed)
            PiLM_outputs = PiLM.generate_text(prefix_text)

            output_text = tokenizer.decode(PiLM_outputs['output_ids'][0])
            print(output_text)

            output_text = PiLM.extract_text(output_text)
            
            colorful_text, eval_metrics = colorful(output_text, args.topic, spacy_model, wordlists)

            print('-'*100)
            print(colorful_text)
            print('-'*50)
            print('[hits] = {}'.format(eval_metrics['hits']))
            print('[hits related] = {}'.format(eval_metrics['hits_related']))
            print('[on-topic] = {}'.format(eval_metrics['on-topic']))
            print('[on-topic related] = {}'.format(eval_metrics['on-topic_related']))
            print()

            obj = {'args': vars(args),
                   'output_text': output_text,


                   'hits':             eval_metrics['hits'],
                   'hits_related':     eval_metrics['hits_related'],
                   'on-topic':         eval_metrics['on-topic'],
                   'on-topic_related': eval_metrics['on-topic_related'],

                   'step_record': PiLM_outputs['step_record'].tolist(),
                   'time_record': PiLM_outputs['time_record'].tolist()}

            if args.output_dir is not None:
                with open(output_path, mode='w', encoding='utf-8') as fp:
                    json.dump(obj=obj, fp=fp, indent=4)
                
                if args.save_latent:
                    with open(output_path.replace('.json', '.pickle'), mode='wb') as fp:
                        pickle.dump(PiLM_outputs['latent_collect'], file=fp)


