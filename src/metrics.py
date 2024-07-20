import torch

def filter_special_token(text):
    text = text.replace('<|endoftext|>', ' ') # GPT
    text = text.replace('</s>', ' ') # OPT
    text = text.strip()
    return text

@torch.no_grad()
def get_PPL(text, model, tokenizer):
    # text = text.replace('<|endoftext|>', ' ') # GPT
    # text = text.replace('</s>', ' ') # OPT
    # text = text.strip()
    text = filter_special_token(text)
    text = tokenizer.unk_token + text

    inputs = tokenizer(text=text,
                       return_tensors='pt').to(model.device)
    input_ids = inputs.input_ids
    labels = input_ids.clone()
    labels[labels==tokenizer.pad_token_id] = -100
    
    entropy = model(input_ids=input_ids,
                    labels=labels).loss
    
    ppl = torch.exp(entropy)
    return ppl.item()

@torch.no_grad()
def get_grammar(text, model, tokenizer):
    text = filter_special_token(text)
    # text = text.replace('<|endoftext|>', ' ') # GPT
    # text = text.replace('</s>', ' ') # OPT
    # text = text.strip()

    inputs = tokenizer(text=text,
                       return_tensors='pt',
                      ).to(model.device)
    
    input_ids = inputs.input_ids
    
    outputs = model(**inputs)
    grammar_score = outputs.logits.softmax(-1).flatten()[1]
    return grammar_score.item()


# def cleanUNK(text, tokenizer):
#     return tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True)

# @torch.no_grad()
# def get_PPL_woUNK(text, model, tokenizer):
#     text = text.replace('<|endoftext|>', ' ') # GPT
#     text = text.replace('</s>', ' ') # OPT
#     text = tokenizer.unk_token + tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True) #remove UNK token in text
#     inputs = tokenizer(text=text,
#                        return_tensors='pt',
#                       ).to(model.device)
    
#     input_ids = inputs.input_ids


#     labels = input_ids.clone()
#     labels[labels==tokenizer.pad_token_id] = -100
    
#     entropy = model(input_ids=input_ids,
#                     labels=labels).loss
    
#     ppl = torch.exp(entropy)
#     return ppl.item()