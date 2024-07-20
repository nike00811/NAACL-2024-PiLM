import numpy as np
import torch
import time
import json
import os
import pickle
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def retype(past):
    if type(past) == np.ndarray:
        return past
    ret = []
    for layer in past:
        if type(layer) == torch.Tensor:
            ret.append(layer)
        else:
            ret.append(torch.stack(layer))
    ret = torch.stack(ret)
    return ret

class Plug_in_LM:
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg

    def get_loss(self,
                 last,
                 perturbed_past,
                 previous_bias,
                 prefix_text,
                 generate_iteration=0,
                 generated_ids=[]):
        # last.shape = [1, 1]
        outputs = self.model(last, past_key_values=perturbed_past)
        first_logits = outputs.logits                                                      # first_logits.shape = [1, 1, 50257]
        first_distribution = first_logits.squeeze().softmax(dim=-1)                        # first_distribution.shape = [50257]
        past_kv = retype(outputs.past_key_values)                                          # past_kv.shape = [num_layer, 2, 1, num_heads, sequence_length, embed_size_per_head]

        sampling = first_distribution.multinomial(self.args.N, replacement=self.args.replacement)    # sampling.shape = [N]
        batch_Yids = sampling.unsqueeze(-1)                                                # batch_last.shape = [N, 1]
        batch_probs = first_distribution[batch_Yids]                                       # batch_probs.shape = [N, 1]
        batch_past_key_values = past_kv.repeat(1, 1, self.args.N, 1, 1, 1)                      # batch_past_key_values.shape = [num_layer, 2, N, num_heads, sequence_length, embed_size_per_head]

        
        for i in range(1, self.args.future_n_tokens):
            outputs = self.model(input_ids=batch_Yids[:, -1:], past_key_values=batch_past_key_values)
            
            logits = outputs.logits                                                        # logits.shape = [N, 1, 50257]
            distribution = logits.squeeze(1).softmax(dim=-1)
            sample_indice = distribution.multinomial(1)                                    # sample_indice.shape = [N, 1]
            
            batch_past_key_values = retype(outputs.past_key_values)
            current_probs = distribution[torch.arange(self.args.N), sample_indice.squeeze()].unsqueeze(-1)
            batch_Yids = torch.concat((batch_Yids, sample_indice), dim=1)
            batch_probs = torch.concat((batch_probs, current_probs), dim=1)
            
            
        # batch_Yids.shape = [N, future_n_tokens]
        # batch_probs.shape = [N, future_n_tokens]
        logprobs = []
        batch_text = []
        for ids, probs in zip(batch_Yids, batch_probs):
            text = self.tokenizer.decode(ids)
            batch_text.append(text)
            
            logp = probs.log()
            logp = logp.sum(-1)                                   # logp.shape = []
            logprobs.append( logp )
        
        
        logprobs = torch.stack(logprobs)
        # logprobs.shape = [N]
        
        rewards = self.get_reward(batch_text, prefix_text).to(self.args.device)
        # rewards.shape = [N]

        if self.args.bias_method == 'mean':
            bias = rewards.mean() # average
        else:
            bias = 0

        
        if self.args.bias_location == 'previous':
            loss = -1 * (rewards - previous_bias) * logprobs
        elif self.args.bias_location == 'current':
            loss = -1 * (rewards - bias) * logprobs
            
        loss = loss.mean()

        # if self.args.debug:
        #     print('rewards.mean() = {}'.format(rewards.mean()))
        #     print('previous_bias = {}'.format(previous_bias))
        #     print('loss = {}'.format(loss))
        
        return loss, bias

    @torch.no_grad()
    def auto_regressive(self, last, pkv, prefix_ids, n):
        osf = prefix_ids[:] # osf = output so far
        next_n_token = []
        for _ in range(n):
            input_ids = torch.tensor(osf, device=self.args.device).unsqueeze(0)
            unpert_outputs = self.model(input_ids)
            unpert_logits = unpert_outputs.logits
            unpert_probs = unpert_logits[0, -1].squeeze().softmax(dim=-1)
            
                    
            outputs = self.model(input_ids=last,
                                 past_key_values=pkv)
            logits = outputs.logits
            pkv = retype(outputs.past_key_values)
            
            pert_probs = logits.squeeze().softmax(dim=-1)
            # pert_probs.shape = [50257]
            
            # Fuse the modified model and original model
            pert_probs = ((pert_probs ** self.args.gm_scale) * (unpert_probs ** (1 - self.args.gm_scale)))  # + SMALL_CONST
            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

            
            if self.args.sample:
                probs, indices = pert_probs.topk(self.args.topk)
                # probs.shape = [self.args.topk]
                # indices.shape = [self.args.topk]
                idx = probs.multinomial(1) # sampling
                last = indices[idx].unsqueeze(0)
            else:
                last = pert_probs.argmax().unsqueeze(0) # greedy
            # last.shape = [[1]]
                
            osf.append( last.item() )
            next_n_token.append( last.item() )

        return next_n_token, pkv

    def extract_text(self, text):
        text = text[self.start_pos:].strip()
        return text

    def check_good_enough(self, text):
        prefix_reward = self.get_reward(batch_text=[text])
        if prefix_reward > 1.99:
            return True
        return False

    def generate_text(self, prefix_text):
        if self.args.use_prompt == 0:
            full_text = prefix_text
            input_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors='pt')
            # prompt_pos = [None, None]
        elif self.args.use_prompt == 1:
            full_text = '{}{}{}'.format(self.args.topic, self.tokenizer.eos_token, prefix_text)
            prompt_pos = [full_text.find(self.args.topic), full_text.find(self.args.topic)]
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors='pt')

            left  = (input_ids == self.tokenizer.encode(self.args.topic)[0]).nonzero()[0, -1]
            right = (input_ids == self.tokenizer.encode(self.args.topic)[-1]).nonzero()[0, -1]
            prompt_pos = {'left': left, 'right': right}
        elif self.args.use_prompt == 2:
            full_text = '(Please write a paragraph regarding {}) {}'.format(self.args.topic, prefix_text)
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors='pt')

            # tokenizer.encode('(')[0] = 7
            # tokenizer.encode(')')[0] = 8
            left  = (input_ids == self.tokenizer.encode('(')[0]).nonzero()[0, -1]
            right = (input_ids == self.tokenizer.encode(')')[0]).nonzero()[0, -1]
            prompt_pos = {'left': left, 'right': right}

        elif self.args.use_prompt == 3:

            full_text = '(This is a paragraph about {}) {}'.format(self.args.topic, prefix_text)
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors='pt')

            # tokenizer.encode('(')[0] = 7
            # tokenizer.encode(')')[0] = 8
            left  = (input_ids == self.tokenizer.encode('(')[0]).nonzero()[0, -1]
            right = (input_ids == self.tokenizer.encode(')')[0]).nonzero()[0, -1]
            prompt_pos = {'left': left, 'right': right}

        self.start_pos = full_text.find(prefix_text)
#         print('[2023-03-07] prompt_pos = {}'.format(prompt_pos))

        output_so_far = input_ids.to(self.args.device)
        # initialization
        # run model forward to obtain unperturbed
        last = output_so_far[:, -1:]
        outputs = self.model(output_so_far[:, :-1])
        past = retype(outputs.past_key_values)

        step_record = []
        time_record = []
        latent_collect = []
        good_enough = False
        for i in range(0, self.args.length, self.args.generate_n_tokens):
            self.remaining_n = self.args.length - i
            if not good_enough:
                full_text = self.tokenizer.decode(output_so_far[0])
                prefix_text = self.extract_text(full_text)
                good_enough = self.check_good_enough(prefix_text)


            if not good_enough:
                if self.args.reset_prefix_hidden:
                    previous_past = past.clone()

                    if self.args.keep_prompt:
                        # print('[2023-03-08] output_so_far.shape = {}'.format(output_so_far.shape))
                        # print('[2023-03-08] output_so_far = {}'.format(output_so_far))
                        # print('[2023-03-08] output_so_far[:, prompt_pos['right']+1:-1].shape = {}'.format(output_so_far[:, prompt_pos['right']+1:-1].shape))
                        # print('[2023-03-08] past[:, :, :, :, :prompt_pos['right']+1, :].shape = {}'.format(past[:, :, :, :, :prompt_pos['right']+1, :].shape))
                        outputs = self.model(input_ids=output_so_far[:, prompt_pos['right']+1:-1],
                                             past_key_values=past[:, :, :, :, :prompt_pos['right']+1, :])
                        past = retype(outputs.past_key_values)

                    else:
                        outputs = self.model(input_ids=output_so_far[:, :-1])
                        past = retype(outputs.past_key_values)
#                     print('previous_past == past = {}'.format(torch.all( previous_past == past )))
#                     print('(prompt) previous_past == past =',
#                            torch.all(previous_past[:, :, :, :, :prompt_pos['right']+1, :] == past[:, :, :, :, :prompt_pos['right']+1, :]))
                    # print('previous_past.shape() = {}'.format(previous_past.shape))
                    # print('past.shape() = {}'.format(past.shape))
                    
                start_time = time.time()
                print('O_______________O prefix text = {}'.format(self.tokenizer.batch_decode(output_so_far)))
                
                
                output_ids, past, loss_this_iter = self.perturb_past(
                                                                     past=past,
                                                                     last=last,
                                                                     generate_iteration=i,
                                                                     output_so_far=output_so_far,
                                                                     latent_collect=latent_collect,
                                                                    )


                cost = time.time()-start_time
                time_record.append(cost)
                step_record.append(len(loss_this_iter))
                print('[INFO] loss_this_iter = {}'.format(np.array(loss_this_iter)))
            else:
                print('[INFO] early stop')
                output_ids, past = self.auto_regressive(last=last,
                                                        pkv=past,
                                                        prefix_ids=output_so_far[0].tolist(),
                                                        n=min(self.args.generate_n_tokens, self.remaining_n))
            
            output_ids = torch.tensor(output_ids, device=self.args.device).unsqueeze(0)
            last = output_ids[:, -1:]
            # last.shape = [[1]]
            
            output_so_far = torch.cat((output_so_far, output_ids), dim=1)
            torch.cuda.empty_cache()
        
        ret = {}
        ret['output_ids'] = output_so_far
        ret['step_record'] = np.array(step_record)
        ret['time_record'] = np.array(time_record)
        ret['latent_collect'] = latent_collect
        return ret


    @torch.no_grad()
    def predict_KV(self, past):
        pred_results = []
        length = past.shape[4]
        for pos in range(length):
            inputs = past[:, :, :, :, pos:pos+1, :].unsqueeze(0).to(self.latent_controller.device)
            output = self.latent_controller(inputs)[0]
            pred_results.append(output)
        pred_results = torch.concat(pred_results, dim=4)
        return pred_results.to(self.model.device)


    def perturb_past(self,
                     past,
                     last,
                     generate_iteration,
                     output_so_far,
                     latent_collect,
                    ):
        generated_ids = output_so_far[0, :].tolist()

        if self.args.use_controller:
            pert_past = self.predict_KV(past)
            latent_collect.append([past, pert_past])
            output_ids, pkv = self.auto_regressive(last, pert_past, generated_ids, min(self.args.generate_n_tokens, self.remaining_n))
            return output_ids, pkv, []


        SMALL_CONST = 1e-15
        past = retype(past)
        before_kv = past.clone().to('cpu')
        
        num_layer = len(past)
        grad_shape = [num_layer, 2] + list(past[0][0].shape)
        # [num_layer, 2, batch_size, num_heads, sequence_length, embed_size_per_head]
        grad_accumulator = torch.zeros(grad_shape, device=self.args.device, dtype=past.dtype)

        loss_per_iter = []
        
        window_mask = torch.ones(past[0].shape, device=self.args.device, dtype=past.dtype)
    #     print('window_mask.shape = {}'.format(window_mask.shape))
        
        freeze_matrix = torch.ones(grad_accumulator.shape, device=self.args.device, dtype=past.dtype)
        for i in range(grad_accumulator.shape[4]-self.args.modify_width):
            freeze_matrix[:, :, :, :, i, :] = 0
        
    #     # fix prompt and <|endoftext|>
    #     eos_index = generated_ids.index(self.tokenizer.eos_token_id)
    #     for i in range(eos_index+1):
    #         freeze_matrix[:, :, :, :, i, :] = 0

    #     print('freeze_matrix.shape = {}'.format(freeze_matrix.shape))
    #     print('freeze_matrix.sum() = {}'.format(freeze_matrix.sum()))
        
        
        full_text = self.tokenizer.decode(generated_ids)
        prefix_text = self.extract_text(full_text)
        bias = 0
        for i in tqdm(range(self.args.M), desc='iteration {:2d}'.format(generate_iteration), bar_format='{l_bar}{bar:30}{r_bar}'):
    #         curr_perturbation = grad_accumulator.to('cuda')
    #         curr_perturbation.requires_grad = True

            curr_perturbation = []
            for layer in range(num_layer):
    #             KV = grad_accumulator[layer].clone()
                KV = grad_accumulator[layer].clone() * freeze_matrix[layer]
                KV.requires_grad = True
                # freeze layer 2023-01-17
    #             if layer > 0:
    #                 KV.requires_grad = False
                curr_perturbation.append(KV)
        
            # curr_perturbation: delta H
            
            # Compute hidden using perturbed past
            perturbed_past = []
            for layer in range(num_layer):
                KV = past[layer] + curr_perturbation[layer]
                perturbed_past.append(KV)
                
            
            # [2023-02-17]
            if self.args.dynamic_M:
                output_ids, pkv = self.auto_regressive(last, perturbed_past, generated_ids, min(self.args.generate_n_tokens, self.remaining_n))
                output_text = self.tokenizer.decode(output_ids)
                score = self.get_reward([output_text], prefix_text)[0]
                self.dynamic_M_threshold = [1.5, 1.6, 1.7, 1.8, 1.9]
                threshold = self.dynamic_M_threshold[generate_iteration//self.args.generate_n_tokens]

                if self.args.debug:
                    print('score, threshold = {:.4f}, {:.4f}'.format(score, threshold))
                
                if score > threshold:
                    return output_ids, pkv, loss_per_iter
                
                
            loss = 0.0        
            RL_loss, bias = self.get_loss(last=last,
                                          perturbed_past=perturbed_past,
                                          previous_bias = bias,
                                          prefix_text = prefix_text,
                                          generate_iteration=generate_iteration,
                                          generated_ids=generated_ids,
                                         )

            loss += RL_loss
            
            loss_per_iter.append(loss.item())

            # compute gradients
            loss.backward()
            
            # calculate gradient norms
            grad_norms = [ (torch.norm(p_.grad*window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation) ]

            # normalize gradients
            grad = [ -self.args.stepsize * (p_.grad * window_mask / grad_norms[index] ** self.args.gamma) for index, p_ in enumerate(curr_perturbation) ]

    #         # freeze layer 2023-01-17
    #         grad = []
    #         for index, p_ in enumerate(curr_perturbation):
    #             if p_.grad is not None:
    #                 grad.append( -self.args.stepsize * (p_.grad * window_mask) )
    #             else:
    #                 grad.append( torch.zeros(p_.shape, device=p_.device) )

            # reset gradients, just to make sure
            for p_ in curr_perturbation:
                if p_.grad is not None:
                    p_.grad.data.zero_()

            # accumulate gradient
            for layer in range(num_layer):
                grad_accumulator[layer] += grad[layer]
            
            # removing past from the graph
            new_past = []
            for p_ in past:
                new_past.append(p_.detach())
            past = new_past
            

    #     grad_accumulator *= freeze_matrix
        pert_past = []
        for layer in range(num_layer):
            KV = past[layer] + grad_accumulator[layer] * freeze_matrix[layer]
            pert_past.append(KV)


        after_kv = retype(pert_past).to('cpu')
        latent_collect.append([before_kv, after_kv])


        output_ids, pkv = self.auto_regressive(last, pert_past, generated_ids, min(self.args.generate_n_tokens, self.remaining_n))
        # output_text = self.tokenizer.decode(output_ids)
        
        return output_ids, pkv, loss_per_iter