#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import numpy as np
from transformers import BertTokenizer


# In[4]:


class QuoraBertDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, text_path='../data/quora_train.txt', tokenizer='bert-base-uncased'):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        self.tokenizer = self.init_tokenizer(tokenizer)
        self.sentences = self.read_text(text_path)
        self.init_constants()
        
        self.n_words = len(self.tokenizer)

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
        
        return (seq1_tensor, seq2_tensor)

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'val':
            return self.val_size
        else:
            return self.test_size

    def init_tokenizer(self, tokenizer):
        if tokenizer == 'bert-base-cased':
            pretrained_model_name = "bert-base-cased"
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(tokenizer)
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
    
    def read_text(self, text_path):
        # add words to dictionary
        f = open(text_path, 'r')
        lines = f.readlines()
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
            

        def normalize_sentence(s):
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            s = s.lower()
            return s
        sentences = []
        for line in tqdm(lines):
            seq1, seq2 = [normalize_sentence(seq) for seq in line.split('\t')]
            sentences.append((seq1, seq2))            
        
        return lines


# In[5]:


# dataset = QuoraBertDataset("train", 124000, 100, text_path='../../data/quora_train.txt')


# In[6]:


# def create_mini_batch(samples):
#     seq1_tensors = [s[0] for s in samples]
#     seq2_tensors = [s[1] for s in samples]

#     # zero pad
#     seq1_tensors = pad_sequence(seq1_tensors,
#                                   batch_first=True)

#     seq2_tensors = pad_sequence(seq2_tensors,
#                                   batch_first=True)    
    
#     return seq1_tensors, seq2_tensors


# # it takes time to predict masked component
# # to improve -> use gpu and calculate outside the dataset
# data_loader = DataLoader(dataset, batch_size=64, collate_fn=create_mini_batch)


# In[7]:


# max_size = 0
# for seq1, seq2 in tqdm(data_loader):
#     pass
#     max_size = max(max_size, seq1.shape[1])
# print(max_size) # 88


# In[ ]:




