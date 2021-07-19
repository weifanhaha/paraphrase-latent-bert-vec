#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.corpus import wordnet 
import random

import re
import numpy as np
from tqdm import tqdm


# In[2]:


# for dictionary
# from quora_dataset import QuoraDataset
from nltk.corpus import stopwords
# called from src
from datasets.quora_dataset import QuoraDataset


# In[6]:


class QuoraWordnetAugDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, text_path='../data/quora_train.txt',
                 word2idx_path='../data/word2idx.npy', idx2word_path='../data/idx2word_path.npy',
                 dic_sentences_num=150000, load_dic=True, replace_prob=0.5):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.replace_prob = replace_prob
        self.sentences = []

        self._init_constants()
        self._init_sentences(text_path)
        self.word2idx, self.idx2word = QuoraDataset.build_dictionary(
            text_path=text_path, sentences_num=dic_sentences_num, load_dic=load_dic,
            word2idx_path=word2idx_path, idx2word_path=idx2word_path
        )
        self.n_words = len(self.word2idx)
        self.stopwords = stopwords.words('english')

    def __getitem__(self, idx):
        seq1, seq2 = self.sentences[idx]
        
        idxes1 = [self._get_index(word) for word in seq1.split(' ')]
        idxes1 = [self.SOS_token_id] + idxes1 + [self.EOS_token_id]
        
#         idxes2 = [self._get_index(word) for word in seq2.split(' ')]
#         idxes2 = [self.SOS_token_id] + idxes2 + [self.EOS_token_id]
        idxes2 = self._get_aug_sentence(seq2)
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

    def _get_aug_sentence(self, seq):
        words = seq.split()
        new_sentence_idxes = []
        
        for word in words:
            syn_set = self._get_synset(word)
            
            if word in self.stopwords or len(syn_set) == 0:
                new_sentence_idxes.append(self._get_index(word))
                continue

            # replace with prob 
            if random.random() > self.replace_prob:
                new_sentence_idxes.append(self._get_index(word))
                continue
            
            sampled_syn = random.sample(syn_set, 1)[0]
            if self._get_index(sampled_syn) != self.UNK_token_id:
                new_sentence_idxes.append(self._get_index(sampled_syn))
            else:
                new_sentence_idxes.append(self._get_index(word))
                
        return new_sentence_idxes
    
    def _get_synset(self, word):
        syn_set = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas(): 
                # ignore the phrase
                if '_' in l.name() or '-' in l.name():
                    continue
                syn_set.add(l.name()) 
        if word in syn_set:
            syn_set.remove(word)
        return syn_set



# In[7]:



# dataset = QuoraWordnetAugDataset("train", 50000, 1000, text_path='../../data/quora_train.txt', load_dic=True,
# word2idx_path='../../data/word2idx.npy', idx2word_path='../../data/idx2word_path.npy')
# # # dataset = QuoraDataset("train", 50000, 1000, load_dic=False)
# # word2idx, idx2word = QuoraDataset.build_dictionary(sentences_num=100000, load_dic=False)



# In[8]:


# repeat = 3
# max_count = 3

# for _ in range(repeat):
#     print('################')
#     count = 0
#     for seq1, seq2 in dataset:
#         words1 =  [dataset.idx2word[idx.item()] for idx in seq1] 
#         words2 = [dataset.idx2word[idx.item()] for idx in seq2]
#         print(' '.join(words1))
#         print(' '.join(words2))
#         print('-------------------')
#         count += 1
#         if count > max_count:
#             break


# In[ ]:




