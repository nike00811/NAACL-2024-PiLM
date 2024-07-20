import json
import pickle
import spacy
import numpy as np
import torch
import os
import argparse

def lemmatization(text):
    tokens = spacy_model(text)
    lemma = ''.join([token.lemma_+token.whitespace_ for token in tokens])
    return lemma

def get_embedding(lemma_text):
    tokens = spacy_model(lemma_text.lower())
    embedding = []
    for token in tokens:
        embedding.append(token.vector)
    # type(embedding) = list of np.array
    embedding = np.array(embedding)
    # type(embedding) = 2D np.array
    embedding = torch.tensor(embedding)
    # type(embedding) = 2D torch.tensor
    return embedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default='en_core_web_lg')
    parser.add_argument("--file",        type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))
    
    spacy_model = spacy.load(args.spacy_model)


    if os.path.exists(args.output_path):
        print('[INFO] {} already exists'.format(args.output_path))
        exit()

    with open(args.file) as reader:
        wordlists = json.load(reader)

    wordlist_embedding = {}
    for topic in wordlists.keys():
        tmp = []
        for word in wordlists[topic]:
            lemma_word = lemmatization(word)
            
            embedding = get_embedding( lemma_word )
            if embedding.shape[0] != 1:
                print('[SKIP] {}: {}'.format(topic, word))
                continue
            tmp.append( embedding )


        wordlist_embedding[topic] = torch.stack(tmp).squeeze(1)
        print('{}: {}'.format(topic, wordlist_embedding[topic].shape))
        print('-'*100)
    with open(args.output_path, mode='wb') as f:
        pickle.dump(wordlist_embedding, f)


