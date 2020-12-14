import numpy as np
import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import pickle
from torch.distributions import Categorical
import torch.nn.functional as F

class Generator():
        def __init__(self, seq_len, str_map):
                self.seq_len = seq_len
                self.vocab_size = len(str_map)

                # declare our model, wanting to see hidden states
                self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True, use_cache=True).cuda()

                # freeze transformer
                for param in self.model.transformer.parameters():
                    param.requires_grad = False

                # mod head for our coco vocab
                self.model.lm_head = nn.Linear(self.model.lm_head.in_features, self.vocab_size).cuda()

                # Just making sure the FC layer is not frozen :)
                for param in self.model.lm_head.parameters():
                    param.requires_grad = True

                # we will use AdamW as the optimizer
                self.loss = nn.CrossEntropyLoss()
                self.optim = AdamW(self.model.parameters(), lr=5e-5)

                # Use the same tok TODO seems useless
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding=True)
                self.tokenizer.pad_token = self.tokenizer.eos_token

                # map to map non-gpt vocab back into strings
                self.str_map = np.array(str_map)

        def generate(self, batch_size, num_batches, inc_hidden_state, inc_probs, decode):
            # put into eval mode
            self.model.eval()

            # placeholder for generated words
            generated = torch.empty(batch_size*num_batches, self.seq_len, dtype=torch.long).cuda()

            # tensor of probabilities
            probs = torch.empty(batch_size*num_batches, self.seq_len, self.vocab_size).cuda()

            # tensor of hidden states
            h_states = torch.empty(batch_size*num_batches, self.seq_len, self.model.config.n_embd).cuda()

            # start token
            tok = 50256 * torch.ones(batch_size*num_batches, dtype=torch.long).cuda()
            attn_mask = torch.ones(batch_size*num_batches, dtype=torch.long).cuda()
            past = None

            # generate sequence
            for i in range(self.seq_len):
                # forward pass + extract data
                res = self.model(input_ids=tok, attention_mask=attn_mask)
                prob, past, h_state = res[0], res[1], res[2][-1]
                
                # pick out most recent token (if inputted > 1 token)
                # TODO: fix this for having other starts than beg token
                if len(prob.shape) == 3:
                    prob = prob[tok_mask]
                    print('HERE: ', h_state.shape, tok_mask)
                    h_state = h_state[tok_mask]

                # Attach hidden state (last layer)
                h_states[:, i, :] = h_state.squeeze(1)

                # concat to probs array
                probs[:, i, :] = F.log_softmax(prob, dim=1)

                # Sample this prob dist for each sentence.
                dist = Categorical(F.softmax(prob, dim=1))

                # Add the new word to all the sentences (in non-gpt vocab)
                generated[:, i] = dist.sample()

                # map to gpt2 vocab
                str_map = self.str_map[generated[:, :i+1].cpu()].tolist()
                gpt_map = self.tokenizer(str_map, padding=True, is_split_into_words=True)
                tok =  torch.tensor(gpt_map['input_ids']).cuda()
                attn_mask = torch.tensor(gpt_map['attention_mask']).cuda()
                tok_mask = torch.cat((torch.arange(batch_size*num_batches).unsqueeze(1).cuda(), attn_mask.argmax(1).unsqueeze(1)), dim=1).tolist()

            # decode=put back to string
            if decode:
                generated = self.str_map[generated.flatten()].reshape(batch_size*num_batches, self.seq_len)
            else:
                generated = np.split(np.array(generated), batch_size, axis=0)

            res = [generated]
            if inc_hidden_state:
                res.append(h_states)
            if inc_probs:
                res.append(probs)

            return res

        # Gets hidden state for inputted data (for rewards)
        def get_hidden_state(self, data):
            self.model.eval()

            # pass thru transformer
            h_state = self.model(input_ids=tok)[2][-1]

            # pick out last token
            if len(h_state.shape) == 3:
                h_state = h_state[:,-1,:].squeeze(1)

            return h_state

        # fine tune new FC layer  using normal transformer opt & train data
        # https://huggingface.co/transformers/custom_datasets.html
        def pretrain_step(self, batch):
            """
            pretrain_step: one step of pretraining

            param: batch 
            """
            self.model.train()

            # get data from batch
            truth, m_in, mask = batch
            mask = mask.flatten()
            truth = truth.flatten()

            # perform one train step
            self.optim.zero_grad() # clear grad

            # need to calc loss manually because of switching vocab (split into multiple tokens)

            # get out prob
            prob = F.softmax(model(input_ids=m_in)[0], dim=-1).view(-1, self.vocab_size)
            prob = prob[mask, :]

            # compute loss & backprop
            loss = self.loss(prob, truth)
            loss.backward()
            self.optim.step()

            # ret loss
            return loss

        def rl_train_step(self, x, rewards_to_go, probs, decay_weight):
            # TODO: Can take in batch_size instead of 1?
            '''
            Parameters
                x : (1, seq_length)
                rewards_to_go : (1, seq_length)
                probs : (1, seq_length, vocab_size)
            '''

            # Put model in train mode
            self.model.train()
            


            loss = 0
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
