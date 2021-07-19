#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.corpus import wordnet 
import random

from tqdm import tqdm
import re
import numpy as np

# for dictionary
# from quora_dataset import QuoraDataset
from nltk.corpus import stopwords
# called from src
from datasets.quora_dataset import QuoraDataset


# In[4]:


class QuoraWordnetDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, text_path='../data/quora_train.txt',
                 word2idx_path='../data/word2idx.npy', idx2word_path='../data/idx2word_path.npy',
                 dic_sentences_num=150000, load_dic=True, indiv_k=5, topk=50, replace_origin=False, append_bow=True):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.indiv_k = indiv_k
        self.topk = topk
        self.replace_origin = replace_origin
        self.append_bow = append_bow
        self.sentences = []

        self._init_constants()
        self._init_sentences(text_path)
        self.word2idx, self.idx2word = QuoraDataset.build_dictionary(
            text_path=text_path, sentences_num=dic_sentences_num, load_dic=load_dic,
            word2idx_path=word2idx_path, idx2word_path=idx2word_path
        )
        self.n_words = len(self.word2idx)
        self.stopwords = stopwords.words('english')
        # set random seed ?

    def __getitem__(self, idx):
        seq1, seq2 = self.sentences[idx]
        
        idxes1 = [self._get_index(word) for word in seq1.split(' ')]
        idxes1 = [self.SOS_token_id] + idxes1 + [self.EOS_token_id]        

        if self.replace_origin:
            idxes1 = self.get_replaced_sentence_idxes(seq1)
        
        elif self.append_bow:
            # append wordnet synonyms
            wordnet_idxes = self.get_wordnet_idxes(seq1)
            idxes1 = idxes1 + wordnet_idxes + [self.EOS_token_id]
                    
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
            return s

        for line in tqdm(lines):
            seq1, seq2 = [normalize_sentence(seq) for seq in line.split('\t')]
            self.sentences.append((seq1, seq2))

    def get_wordnet_idxes(self, seq):
        indiv_k = self.indiv_k
        words = seq.split()
        wordnet_idxes = []

        for word in words:
            syn_set = self._get_synset(word)
            
            # need not to append
            if len(syn_set) == 0:
                continue

            if len(syn_set) > indiv_k:
                syn_set = random.sample(syn_set, indiv_k)

            for syn in syn_set:
                if self._get_index(syn) != self.UNK_token_id:
                    wordnet_idxes.append(self._get_index(syn))
            if len(wordnet_idxes) > self.topk:
                wordnet_idxes = random.sample(wordnet_idxes, self.topk)            
        return wordnet_idxes

    
    # to be implement
    def get_replaced_sentence_idxes(self, seq):
        indiv_k = self.indiv_k
        words = seq.split()
        new_sentence_idxes = []
        wordnet_idxes = []
        
        for word in words:
            syn_set = self._get_synset(word)
            
            if word in self.stopwords or len(syn_set) == 0:
                new_sentence_idxes.append(self._get_index(word))
                continue
            
            sampled_syn = random.sample(syn_set, 1)[0]
            if self._get_index(sampled_syn) != self.UNK_token_id:
                new_sentence_idxes.append(self._get_index(sampled_syn))
                # append the original word to the bow
                wordnet_idxes.append(self._get_index(word))
            else:
                new_sentence_idxes.append(self._get_index(word))

            if len(syn_set) > indiv_k:
                syn_set = random.sample(syn_set, indiv_k)

            for syn in syn_set:
                if self._get_index(syn) != self.UNK_token_id:
                    wordnet_idxes.append(self._get_index(syn))
            if len(wordnet_idxes) > self.topk:
                wordnet_idxes = random.sample(wordnet_idxes, self.topk)
                
        if self.append_bow:
            return [self.SOS_token_id] + new_sentence_idxes + [self.EOS_token_id] + wordnet_idxes + [self.EOS_token_id]
        else:
            return [self.SOS_token_id] + new_sentence_idxes + [self.EOS_token_id]
    
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


# In[11]:


# dataset = QuoraWordnetDataset("train", 50000, 1000, text_path='../../data/quora_train.txt', load_dic=True,
# word2idx_path='../../data/word2idx.npy', idx2word_path='../../data/idx2word_path.npy', replace_origin=True, append_bow=False)
# # # dataset = QuoraDataset("train", 50000, 1000, load_dic=False)
# # word2idx, idx2word = QuoraDataset.build_dictionary(sentences_num=100000, load_dic=False)


# In[8]:


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
# data_loader = DataLoader(dataset, batch_size=6, collate_fn=create_mini_batch)
# seq1, seq2 = next(iter(data_loader))
# print(seq1, seq2)


# In[6]:


# # analyze
# # sentence = 'What is the food you can eat every day for breakfast lunch and dinner ?'
# sentence = 'How do you speak in front of large groups of people ?'


# words = sentence.split()

# stopws =  stopwords.words('english')

# # indiv_k
# indiv_k = 5
# appends = []

# new_sentence = []

# for word in words:
#     print(word)
    
#     syn_set = set()
    
#     for syn in wordnet.synsets(word):
#         for l in syn.lemmas(): 
#             # ignore the phrase
#             if '_' in l.name() or '-' in l.name():
#                 continue
#             syn_set.add(l.name()) 
#     if word in syn_set:
#         syn_set.remove(word)
#     print(syn_set)
    
#     # need not to append
#     if word in stopws or len(syn_set) == 0:
#         if word in stopws:
#             print("''{}'' is stopword".format(word))
#         new_sentence.append(word)
#         continue
        
#     sampled_syn = random.sample(syn_set, 1)[0]
#     new_sentence.append(sampled_syn)
#     if len(syn_set) > indiv_k:
#         syn_set = random.sample(syn_set, indiv_k)

#     for syn in syn_set:
#         appends.append(syn)
#     # append original word
#     appends.append(word)
# print(appends)
# print(new_sentence)


# In[ ]:




