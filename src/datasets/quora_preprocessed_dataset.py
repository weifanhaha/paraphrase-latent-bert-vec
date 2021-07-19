#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer


# In[9]:


class QuoraPreprocessedDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, preprocessed_file='', tokenizer='bert-base-cased'):
        assert mode in ["train", "val", "test"]
        assert preprocessed_file != ''
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.preprocessed_file = preprocessed_file
        
        self.tokenizer = self.init_tokenizer(tokenizer)
        self.init_constants()
        self.load_preprocessed_file()
        self.n_words = len(self.tokenizer)
        
        self.n_words = len(self.tokenizer)

    def __getitem__(self, idx):
        return self.tensors[idx]

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
    
    def load_preprocessed_file(self):
        tensors = np.load(self.preprocessed_file, allow_pickle=True)

        if self.mode == "train":
            tensors = tensors[:self.train_size]
        elif self.mode == 'val':
            tensors = tensors[self.train_size:self.train_size+self.val_size]
        else:
            tensors = tensors[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        self.tensors = tensors


# In[27]:


# dataset = QuoraPreprocessedDataset("train", 124000, 100, preprocessed_file='../../data/preprocess_all_dpng_transformer_bert_tokenizer_bow_indivtopk.npy')


# In[28]:


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


# In[30]:


# from tqdm import tqdm
# for seq1, seq2 in tqdm(data_loader):
#     pass


# In[ ]:




