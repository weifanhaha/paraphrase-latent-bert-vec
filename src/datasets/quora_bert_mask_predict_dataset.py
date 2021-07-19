#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from random import random

# from bow_strategy import get_bow
from datasets.bow_strategy import get_bow


# In[13]:


class QuoraBertMaskPredictDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, 
                 text_path='../data/quora_train.txt', pretrained_model_name='bert-base-cased', 
                 topk=50, bow_strategy='simple_sum', indiv_topk=10, indiv_topp=0.01, 
                 only_bow=False, use_origin=False, replace_predict=False, append_bow=True, replace_p=0.15):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.topk = topk
        self.bow_strategy = bow_strategy # simple_sum, mask_sum, indiv_topk, indiv_topp, indiv_neighbors
        self.indiv_topk = indiv_topk
        self.indiv_topp = indiv_topp
        self.only_bow = only_bow
        self.use_origin = use_origin
        self.replace_predict = replace_predict
        self.append_bow = append_bow
        self.replace_p = replace_p
        
        self.tokenizer = self.init_tokenizer(pretrained_model_name)
        self.mask_predict_model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.sentences = self.read_text(text_path)
        self.init_constants()
        
        self.n_words = len(self.tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_predict_model = self.mask_predict_model.to(self.device)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        seq1, seq2 = sentence.split('\t')
        
        tokens1 = self.tokenizer.tokenize(seq1)
        word_pieces1 =  [self.SOS_token] + tokens1 + [self.EOS_token]
        idxes1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        
        tokens2 = self.tokenizer.tokenize(seq2)
        word_pieces2 = [self.SOS_token] + tokens2 + [self.EOS_token]
        idxes2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        
        seq1_tensor = torch.tensor(idxes1, dtype=torch.long)
        seq2_tensor = torch.tensor(idxes2, dtype=torch.long)
        
        # mask each word from the begining to the end
        # get the probability distribution of the mask tokens
        pred = self.get_mask_pred_probs(seq1_tensor)
        
        # get the predicted bag of words with the probability distribution above
        bow_tokens_tensor = get_bow(self.bow_strategy, self.n_words, pred, self.topk, self.indiv_topk)
        
        ret_seq1_tensor = seq1_tensor
        if self.replace_predict:
            replaced_sentence_tensor = self.get_replaced_sentence(seq1_tensor, pred)
            
            if not self.append_bow:
                return replaced_sentence_tensor, seq2_tensor
            
            ret_seq1_tensor = self.get_concat_replaced_tensor(
                tokens1, bow_tokens_tensor, replaced_sentence_tensor
            )
            return ret_seq1_tensor, seq2_tensor

        if self.only_bow:
            ret_seq1_tensor = bow_tokens_tensor
            if self.use_origin:
                origin_tokens = self.tokenizer.convert_tokens_to_ids(tokens1)
                origin_tensors = torch.tensor(origin_tokens, dtype=torch.long)
                ret_seq1_tensor = torch.sort(torch.cat((origin_tensors, bow_tokens_tensor)))[0]
            
        else:
            ret_seq1_tensor = torch.cat((seq1_tensor, bow_tokens_tensor, torch.tensor([self.EOS_token_id])))

        return ret_seq1_tensor, seq2_tensor

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'val':
            return self.val_size
        else:
            return self.test_size
        
    # [CLS]  [M]  w2  w3  [SEP]        
    # [CLS]  w1  [M]  w3  [SEP]        
    # [CLS]  w1 w2  [M]  [SEP]        
    # get the probability of [M] for each mask-prediction case
    # mask for (number of words) times and every time get (number of words + 2) probability
    # TODO: get the pred for only once
    def get_mask_pred_probs(self, seq1):
        mask_sentences = []
        
        for i in range(1, len(seq1) - 1):
            mask_seq = seq1.detach().clone()
            mask_seq[i] = self.MASK_token_id
            mask_sentences.append(mask_seq)

        mask_stack = torch.stack(mask_sentences)
        mask_stack = mask_stack.to(self.device)
        
        self.mask_predict_model.eval()

        with torch.no_grad():
            pred = self.mask_predict_model(mask_stack)[0]
        pred = pred.cpu()
        return pred

    def get_replaced_sentence(self, seq1, pred):
        softmax = torch.nn.Softmax(dim=0)
        # add BOS and EOS
        pred_ws = [seq1[0].item()]
        for i in range(pred.shape[0]):
            if random() < self.replace_p:
            # 1. top1, when prob > 0.5
    #         prob, idx = torch.topk(softmax(pred[i][i+1]), 1)
    #         w = idx.item() if prob > 0.5 else seq1[i+1].item() 

    #         # 2. top1
    #         prob, idx = torch.topk(softmax(pred[i][i+1]), 1)
    #         w = idx.item()
    
                # 3. sample
                idx = torch.multinomial(softmax(pred[i][i+1]), 1)[0]
                w = idx.item()

                pred_ws.append(w)
            else:
                pred_ws.append(seq1[i+1])

        pred_ws.append(seq1[-1].item())
        return torch.tensor(pred_ws, dtype=torch.long)
    
    
    def get_concat_replaced_tensor(self, tokens1, bow_tokens_tensor, replaced_sentence_tensor):
        # use original tokens by default
        origin_tokens = self.tokenizer.convert_tokens_to_ids(tokens1)
        origin_tensors = torch.tensor(origin_tokens, dtype=torch.long)
        append_tensors = torch.sort(torch.cat((origin_tensors, bow_tokens_tensor)))[0]
        tensors = torch.cat((replaced_sentence_tensor, append_tensors))

        return tensors
            
        
    def init_tokenizer(self, pretrained_model_name):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        return tokenizer
    
    
    def init_constants(self):
        PAD_id,  SOS_id, EOS_id, UNK_id = self.tokenizer.convert_tokens_to_ids(["[PAD]", "[CLS]", "[SEP]", "[UNK]"])
        self.PAD_token_id = PAD_id
        self.SOS_token_id = SOS_id
        self.EOS_token_id = EOS_id
        self.UNK_token_id = UNK_id
        
        self.PAD_token = '[PAD]'
        self.SOS_token = '[CLS]'
        self.EOS_token = '[SEP]'
        self.UNK_token = '[UNK]'
        
        self.MASK_token = '[MASK]'
        self.MASK_token_id = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        
    def read_text(self, text_path):
        # add words to dictionary
        f = open(text_path, 'r')
        lines = f.readlines()
        # shuffle
        np.random.shuffle(lines)
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        return lines


# In[14]:


# dataset = QuoraBertMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt')
# dataset = QuoraBertMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt', bow_strategy='simple_sum')
# dataset = QuoraBertMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt', bow_strategy='indiv_topk')
# dataset = QuoraBertMaskPredictDataset("train", 124000, 100, text_path='../../data/quora_train.txt', bow_strategy='indiv_neighbors')
# dataset = QuoraBertMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt', bow_strategy='indiv_topk', only_bow=True, use_origin=True)
# dataset = QuoraBertMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt', bow_strategy='indiv_topk', replace_predict=True, append_bow=False)


# In[23]:


# idxs, idxs2 = dataset[1]
# tokens = dataset.tokenizer.convert_ids_to_tokens(idxs)
# tokens2 = dataset.tokenizer.convert_ids_to_tokens(idxs2)

# print(tokens)
# print(tokens2)

# print(dataset.tokenizer.convert_tokens_to_string(tokens))
# print(dataset.tokenizer.convert_tokens_to_string(tokens2))


# In[17]:


# def create_mini_batch(samples):
#     seq1_tensors = [s[0] for s in samples]
#     seq2_tensors = [s[1] for s in samples]
# #     bows_tensors = [s[2] for s in samples]

#     # zero pad
#     seq1_tensors = pad_sequence(seq1_tensors,
#                                   batch_first=True)

#     seq2_tensors = pad_sequence(seq2_tensors,
#                                   batch_first=True)    
    
# #     return seq1_tensors, seq2_tensors, torch.stack(bows_tensors)
#     return seq1_tensors, seq2_tensors


# # it takes time to predict masked component
# # to improve -> use gpu and calculate outside the dataset
# data_loader = DataLoader(dataset, batch_size=64, collate_fn=create_mini_batch)


# In[527]:


# seq1, seq2 = next(iter(data_loader))


# In[7]:


# from tqdm import tqdm
# max_size = 0
# for seq1, seq2 in tqdm(data_loader):
# #     pass
# #     print(seq1)
#     print(seq1.shape, seq2.shape)


# In[8]:


# seq1, seq2 = next(iter(data_loader))
# dataset.tokenizer.convert_ids_to_tokens(seq1[10])


# In[ ]:


# # analyze
# dataset = QuoraBertMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt')


# In[27]:


# sentence = 'How do you speak in front of large groups of people ?'
# tokens1 = dataset.tokenizer.tokenize(sentence)
# word_pieces1 =  [dataset.SOS_token] + tokens1 + [dataset.EOS_token]
# idxes1 = dataset.tokenizer.convert_tokens_to_ids(word_pieces1)
# seq1_tensor = torch.tensor(idxes1, dtype=torch.long)

# def get_mask_pred_probs(seq1):
#         mask_sentences = []
        
#         for i in range(1, len(seq1) - 1):
#             mask_seq = seq1.detach().clone()
#             mask_seq[i] = dataset.MASK_token_id
#             mask_sentences.append(mask_seq)

#         mask_stack = torch.stack(mask_sentences)
#         mask_stack = mask_stack.to(dataset.device)
        
#         dataset.mask_predict_model.eval()

#         with torch.no_grad():
#             pred = dataset.mask_predict_model(mask_stack)[0]
#         pred = pred.cpu()
#         return pred


# pred = get_mask_pred_probs(seq1_tensor)

# # get top 5
# softmax = torch.nn.Softmax(dim=0)
# for i in range(pred.shape[0]):
#     prob, idx = torch.topk(softmax(pred[i][i+1]), 5)
#     print(dataset.tokenizer.convert_ids_to_tokens([seq1_tensor[i+1]]))
#     print(dataset.tokenizer.convert_ids_to_tokens(idx))


# In[ ]:




