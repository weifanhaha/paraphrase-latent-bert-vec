#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
from torch.utils.data import Dataset
import re
import numpy as np

class QuoraTextDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, text_path='../data/quora_train.txt'):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.sentences = []

        self._init_sentences(text_path)

    def __getitem__(self, idx):
        return self.sentences[idx]
        seq1, seq2 = self.sentences[idx]

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'val':
            return self.val_size
        else:
            return self.test_size
    
    def _init_sentences(self, text_path):
        f = open(text_path, 'r')
        lines = f.readlines()
        np.random.shuffle(lines)
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        def normalize_sentence(s):
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            return s

        for line in lines:
            seq1, seq2 = [normalize_sentence(seq) for seq in line.split('\t')]
            self.sentences.append((seq1, seq2))

        


# In[12]:


# dataset = QuoraTextDataset("test", 100000, 4000, 2000, text_path='../../data/quora_train.txt')
# print(dataset[1])


# In[ ]:




