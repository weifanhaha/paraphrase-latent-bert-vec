#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import numpy as np


# In[3]:


class QuoraDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, text_path='../data/quora_train.txt', 
                 dic_sentences_num=150000, load_dic=True):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.sentences = []

        self._init_constants()
        self._init_sentences(text_path)
        self.word2idx, self.idx2word = QuoraDataset.build_dictionary(
            text_path=text_path, sentences_num=dic_sentences_num, load_dic=load_dic
        )
        self.n_words = len(self.word2idx)

    def __getitem__(self, idx):
        seq1, seq2 = self.sentences[idx]
        
        idxes1 = [self._get_index(word) for word in seq1.split(' ')]
        idxes1 = [self.SOS_token_id] + idxes1 + [self.EOS_token_id]
        
        idxes2 = [self._get_index(word) for word in seq2.split(' ')]
        idxes2 = [self.SOS_token_id] + idxes2 + [self.EOS_token_id]

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
    
    def _get_index(self, word):
        try: 
            index = self.word2idx[word]
        except KeyError:
            index = self.UNK_token_id
        return index

    def _init_constants(self):
        self.PAD_token = '<PAD>'
        self.SOS_token = '<SOS>'
        self.EOS_token = '<EOS>'
        self.UNK_token = '<UNK>'
        self.PAD_token_id = 0
        self.SOS_token_id = 1
        self.EOS_token_id = 2
        self.UNK_token_id = 3
    
    def _init_sentences(self, text_path):
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
        
        def normalize_sentence(s):
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            s = s.lower()

            return s

        for line in tqdm(lines):
            seq1, seq2 = [normalize_sentence(seq) for seq in line.split('\t')]
            self.sentences.append((seq1, seq2))

    @staticmethod
    def build_dictionary(
        sentences_num=10000, text_path='../data/quora_train.txt', 
        word2idx_path='../data/word2idx.npy', idx2word_path='../data/idx2word_path.npy', load_dic=True
    ):
        if (load_dic):
            try:
                print("[Info] Loading the Dictionary...")
                word2idx = np.load(word2idx_path, allow_pickle=True).item()
                idx2word = np.load(idx2word_path, allow_pickle=True).item()
                print("[Info] Dictionary Loaded")
                return word2idx, idx2word
            except FileNotFoundError :
                print("[Info] Saved dictionary not found. Initialize the Dictionary...")

        # init constances
        PAD_token = '<PAD>'
        SOS_token = '<SOS>'
        EOS_token = '<EOS>'
        UNK_token = '<UNK>'
        PAD_token_id = 0
        SOS_token_id = 1
        EOS_token_id = 2
        UNK_token_id = 3
        
        # init dictionary
        word2idx = {
            SOS_token: SOS_token_id, 
            EOS_token: EOS_token_id,
            PAD_token: PAD_token_id,
            UNK_token: UNK_token_id
        }
        idx2word = {
            SOS_token_id: SOS_token, 
            EOS_token_id: EOS_token,
            PAD_token_id: PAD_token,
            UNK_token_id: UNK_token
        }
        word2count = {}
        n_words = 4 

        # add words to dictionary
        f = open(text_path, 'r')
        lines = f.readlines()
        lines = lines[:sentences_num]

        def normalize_sentence(s):
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            s = s.lower()
            return s

        for line in tqdm(lines):
            seq1, seq2 = [normalize_sentence(seq) for seq in line.split('\t')]
            for word in seq1.split(' ') + seq2.split(' '):
                if word not in word2idx:
                    word2idx[word] = n_words
                    word2count[word] = 1
                    idx2word[n_words] = word
                    n_words += 1
                else:
                    word2count[word] += 1

        np.save(word2idx_path, word2idx)
        np.save(idx2word_path, idx2word)
        print("[Info] Saved the dictionary")
        return word2idx, idx2word
        
        


# In[3]:


# word2idx, idx2word = QuoraDataset.build_dictionary(sentences_num=100000, load_dic=False)


# In[4]:


# def create_mini_batch(samples):
#     seq1_tensors = [s[0] for s in samples]
#     seq2_tensors = [s[1] for s in samples]

#     # zero pad
#     seq1_tensors = pad_sequence(seq1_tensors,
#                                   batch_first=True)

#     seq2_tensors = pad_sequence(seq2_tensors,
#                                   batch_first=True)    
    
#     return seq1_tensors, seq2_tensors


# # dataset = QuoraDataset("train", 50000, 1000, load_dic=False)
# # data_loader = DataLoader(dataset, batch_size=1, collate_fn=create_mini_batch)
# # seq1, seq2 = next(iter(data_loader))


# In[5]:


# dataset = QuoraDataset("train", 124000, 1000, load_dic=False)
# data_loader = DataLoader(dataset, batch_size=128, collate_fn=create_mini_batch)


# In[7]:


# from tqdm import tqdm
# max_size = 0
# for seq1, seq2 in tqdm(data_loader):
#     max_size = max(max_size, seq1.shape[1])
# print(max_size) # 60


# In[ ]:




