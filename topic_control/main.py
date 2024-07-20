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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model",       type=str,   default="gpt2-medium", help="pretrained model name or path to local checkpoint", )
    parser.add_argument("--pretrained_tokenizer",   type=str,   default="gpt2-medium", help="pretrained tokenizer name or path to local checkpoint", )
    parser.add_argument('--embedding_model',        type=str,   default="en_core_web_lg")
    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument('--data_dir', type=str, default="../data/word_list/")
    parser.add_argument('--prefix_set', type=str, default="../data/prefixes/topic_prefixes.json")
    parser.add_argument("--length",                 type=int,   default=50)
    parser.add_argument("--stepsize",               type=float, default=0.09)
    parser.add_argument('--M',                      type=int,   default=3, help='number of update hidden state')
    parser.add_argument('--N',                      type=int,   default=100, help='number of sampling in RL')
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
#     parser.add_argument('--use_prefix_reward', action='store_true')
    parser.add_argument('--reset_prefix_hidden', action='store_true')
    parser.add_argument('--dynamic_M', action='store_true')
    parser.add_argument('--debug',         action="store_true")
    parser.add_argument('--replacement',  action="store_true")
    parser.add_argument('--use_prompt',   type=int, default=0)
    parser.add_argument('--keep_prompt',     action='store_true')

    
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




class PiLM_topic(Plug_in_LM):
    def __init__(self, args, model, tokenizer, spacy_model, topic_embedding):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.spacy_model = spacy_model
        self.topic_embedding = topic_embedding

        self.get_cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        # self.reward_topk = max(int(args.future_n_tokens * 0.2), 1)
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




def add_color(text, color_code):
    text = '{}{}{}'.format(color_code, text, colorama.Style.RESET_ALL)
    return text

def colorful(text, topic, spacy_model, wordlists):
    tokens = spacy_model(text)
    
    metrics = {}
    metrics['hits'] = []
    metrics['hits_related'] = []
    metrics['on-topic'] = []
    metrics['on-topic_related'] = []
    

    for token in tokens:
        if token.text.lower() in wordlists['heldout_bag'][topic]:
            metrics['on-topic'].append(token.text.lower())
            metrics['on-topic_related'].append(token.lemma_.lower())
            
        elif token.lemma_.lower() in wordlists['heldout_bag_related'][topic]:
            metrics['on-topic_related'].append(token.lemma_.lower())
        
        if token.text.lower() in wordlists['topic_dictionary'][topic]:
            metrics['hits'].append(token.text.lower())
            metrics['hits_related'].append(token.lemma_.lower())
        
        elif token.lemma_.lower() in wordlists['related_dictionary'][topic]:
            metrics['hits_related'].append(token.lemma_.lower())


    output = ''
    for token in tokens:
        if token.text.lower() in wordlists['heldout_bag'][topic]:
            output += add_color(token.text, colorama.Fore.YELLOW)
            
        elif token.lemma_.lower() in wordlists['heldout_bag_related'][topic]:
            output += add_color(token.text, '\x1b[93m')
            
        elif token.text.lower() in wordlists['topic_dictionary'][topic]:
            output += add_color(token.text, colorama.Fore.RED)
            
        elif token.lemma_.lower() in wordlists['related_dictionary'][topic]:
            output += add_color(token.text, '\x1b[91m')
        else:
            output += token.text
        output += token.whitespace_

    return output, metrics




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
    
    spacy_model = spacy.load(args.embedding_model)
    
    if args.use_controller:
        latent_controller = torch.load(args.controller).to(args.device)
        latent_controller.eval()
    else:
        latent_controller = None
    
    models = {}
    models['model'] = model
    models['tokenizer'] = tokenizer
    models['spacy_model'] = spacy_model
    models['latent_controller'] = latent_controller
    
    return models




def load_wordlists(data_dir):
    ret = {}
    with open('{}/topic_words.json'.format(data_dir), mode='r', encoding='utf-8') as reader:
        ret['topic_dictionary'] = json.load(reader)

    with open('{}/topic_words-lemma.json'.format(data_dir), mode='r', encoding='utf-8') as reader:
        ret['related_dictionary'] = json.load(reader)
        
    with open('{}/heldout_bags.json'.format(data_dir), mode='r', encoding='utf-8') as reader:
        ret['heldout_bag'] = json.load(reader)
    
    with open('{}/heldout_bags-lemma.json'.format(data_dir), mode='r', encoding='utf-8') as reader:
        ret['heldout_bag_related'] = json.load(reader)
    
    
    for key in ret.keys():
        for topic in ret[key]:
            tmp = []
            for word in ret[key][topic]:
                if word.lower() not in ret[key][topic]:
                    tmp.append(word.lower())
            ret[key][topic] += tmp
            

    with open('{}/topic_words_embeddings.pickle'.format(data_dir), mode='rb') as f:
        ret['topic_embedding'] = pickle.load(f)
    
    return ret









if __name__ == '__main__':
    args = parse_args()
    print(args)
        
    with open(args.prefix_set) as reader:
        prefixes_set = json.load(reader)['prefix']
        
    topic_attribute = ['Science', 'Space', 'Politics', 'Military', 'Religion', 'Computers', 'Legal']
        
    
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
        model = models['model']
        tokenizer = models['tokenizer']
        spacy_model = models['spacy_model']

    wordlists = load_wordlists(args.data_dir)

    PiLM = PiLM_topic(
                      args=args,
                      model=model,
                      tokenizer=tokenizer,
                      spacy_model=spacy_model,
                      topic_embedding=wordlists['topic_embedding'],
                     )


    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

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

            print('Topic: {}'.format(args.topic))
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

